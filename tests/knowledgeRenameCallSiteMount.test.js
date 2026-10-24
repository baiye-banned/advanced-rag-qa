// Knowledge.vue 重命名入口的**调用点胶水**挂载用例：真 mount 视图、点真按钮走完一次重命名。
//
// 由来（issue #180 验收 1）：`src/views/Knowledge.vue` 的 rename 分支
// （`:423` 的空选中短路、`:436` 的名称预填、`:457` 的同名短路、`:494-502` 的
// 重命名请求 + upsert + 关窗 + 刷新）此前**零前端执行用例**——全仓 `git grep rename tests/`
// 只命中的是一个桩字符串（`chatStore.test.js`）与四份后端用例。这一段"只有阅读证据"。
//
// 本文件要钉的是两条**成功后**的调用点行为，也就是验收原文点名的那两条：
//   1. 重命名成功后侧栏与当前选中库同步（store 的 upsert 真的写进去了，
//      且 `currentKnowledgeBaseId` 仍然指向同一个库）
//   2. 刷新取数失败时 `refreshKnowledgeBasesPreserving` 的 upsert 回退生效，
//      并且**不产生未处理拒绝**（`fetchKnowledgeBases()` 被包在 try/catch 里，
//      catch 只 upsert、不重新抛出，所以取数失败不得外泄成 unhandledrejection）
//
// 与既有用例的分工（互不重叠）：
//   tests/knowledgeViewWiring.test.js  静态读文件，证明「视图里写了这几行」
//   tests/knowledgeFeedback.test.js    注入替身测纯函数本体的三态返回
//   本文件                              视图真的跑起来之后，那几行有没有生效
//
// 替身只扎在模块边界 `api/request.js` 上（同 knowledgeDeleteCallSiteMount.test.js）：
// knowledgeAPI、pinia 的 knowledge store、fileListRequest 与视图本身全是真货，
// 所以「侧栏与选中同步」是组件真实响应式状态驱动的 DOM 结果，不是断言一个被 stub 的开关。
//
// 本文件对 `src/` **零改动**（issue #180 验收 6）：用例跑的是 tip 上的产品代码，
// 没有为「让用例变绿」而动过任何一个 src 文件。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { createTestRouter } from './helpers/testRouter.js'
import { calls, resetRequestStub, respond } from './helpers/stubApiRequest.js'

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
}

const KB1 = { id: 'kb1', name: 'KB1' }
const KB2 = { id: 'kb2', name: 'KB2' }
const ONE_FILE = [{ id: 'f1', name: 'a.pdf', size: 10, created_at: '2024-01-01T00:00:00Z' }]
const NAME_INPUT = 'input[placeholder="请输入知识库名称"]'
// 「当前选中库」的判据：模板 :11 的知识库选择器（唯一带 w-48 的那个 el-select），
// Element Plus 把选中项渲染在 .el-select__selected-item 里，文案即 `${name}（${file_count}）`。
// 不取整个 .el-select 的 textContent：下拉里的每个 el-option 也在同一棵子树里，
// 那样「列表里有这个库」和「当前选中的就是这个库」会被混成一条断言。
const CURRENT_BASE_LABEL = '.el-select.w-48 .el-select__selected-item'

// 重命名链的 await 层数是「点击 -> 表单校验 -> PUT -> 成功提示 -> 关窗 -> 刷新知识库列表
// -> 取数 -> 响应」，vueMount 默认的 3 轮不够深：浅了断言会看到「还没发生」的假绿
//（尤其是「对话框必须已关」这类负向断言）。
//
// 光加轮数不够，还得给真实时间：flush 的每一轮都是 microtask + setTimeout(0)，
// 而对话框关闭走的是 Element Plus 的 leave 过渡，落地在 jsdom 的 requestAnimationFrame 上
//（约 16ms）。与 knowledgeCreateCallSiteMount.test.js 同一个理由。
const settle = async (view) => {
  await view.flush(8)
  await new Promise((resolve) => setTimeout(resolve, 40))
}

// 等一个可观测条件成立，超时就把当前状态原样交给后面的断言去报。
async function until(view, predicate, budgetMs = 1000) {
  const deadline = Date.now() + budgetMs
  for (;;) {
    if (predicate()) return true
    if (Date.now() > deadline) return false
    await view.flush(1)
    await new Promise((resolve) => setTimeout(resolve, 2))
  }
}

function requestsSince(mark) {
  return calls.slice(mark).map((entry) => `${entry.method} ${entry.args[0]}`)
}

// 当前选中库的文案（读不到元素时返回 null，让断言报「选择器没命中」而不是静默通过）。
//
// 取「最后一个非空」的 .el-select__selected-item，而不是 querySelector 的第一个：
// Element Plus 2.14 在同一个选择器里放了两个 —— 先是 .el-select__input-wrapper
//（隐藏的搜索输入框，textContent 为空），后是真正显示选中项的那个。
// 取第一个会稳定读到空串，症状是断言报 '' !== 'KB1（0）'，看着像「选中丢了」。
function currentBaseLabel(view) {
  const items = view
    .queryAll(CURRENT_BASE_LABEL)
    .map((node) => node.textContent.trim())
    .filter(Boolean)
  return items.length ? items[items.length - 1] : null
}

function dialogOpen(view) {
  return Boolean(view.query(NAME_INPUT))
}

// 一次挂载 = 一次完整的视图启动（onMounted 会拉知识库列表与文件列表）。
//   rename               PUT /knowledge-bases/:id 的处理器；不传时按「服务端改名成功」返回
//   listErrorAfterRename 非空时，**重命名之后**那次 GET /knowledge-bases 抛这个错误
//
// 取数失败必须只打在重命名之后：挂载期那次也失败的话，视图连当前库都选不出来，
// 用例红的就不是被测的那条断言了。
async function mountKnowledge({ bases = [KB1, KB2], rename, listErrorAfterRename } = {}) {
  resetRequestStub()

  const world = { bases: [...bases] }
  let renamed = false

  respond('get', (url) => {
    if (url === '/knowledge-bases') {
      if (listErrorAfterRename && renamed) throw new Error(listErrorAfterRename)
      return world.bases
    }
    if (url === '/knowledge') return ONE_FILE
    return {}
  })
  respond('put', (url, body) => {
    renamed = true
    if (rename) return rename(url, body)
    // 服务端真的改了名：刷新拿到的列表里应当是改名后的对象。
    const id = String(url).split('/').pop()
    world.bases = world.bases.map((base) => (base.id === id ? { ...base, name: body.name } : base))
    return { id, name: body.name }
  })

  const router = await createTestRouter()
  const view = await mountSfc('views/Knowledge.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  // 挂载期有两次 GET（知识库列表 + 文件列表）。等它们都落地再让调用方划水位线，
  // 否则迟到的挂载请求会混进「点击之后发了什么」里。
  await until(view, () => calls.filter((entry) => entry.method === 'get').length >= 2)
  return view
}

// 把当前知识库切成 name 那个（走真实的 el-select：点开下拉、点选项）。
//
// 必须先把当前库从列表第一个挪走，后面那条「重命名后当前选中库同步」才有承重：
// 重命名 kb1（列表第一个）时，哪怕代码把选中重置回 `bases[0]`，看到的也还是 kb1，
// 断言恒真。切到 kb2 之后，「选中掉回第一个」才会以 `KB1（0）` 的形态暴露出来。
//
// 选项不会渲染在 host 里：el-select 的下拉是 teleport 到 body 的 popper，
// 而且同一页三个 el-select 会一起展开（8 个候选项混在一起），所以按 aria-controls
// 指向的那个 list 精确取，免得点到别的选择器的选项、或点到上一个用例残留的 popper。
async function switchBase(view, label) {
  const wrapper = view.query('.el-select.w-48 .el-select__wrapper')
  assert.ok(wrapper, '页面上应当有知识库选择器')
  wrapper.dispatchEvent(new MouseEvent('click', { bubbles: true }))
  await until(view, () => Boolean(optionsOf(view)))
  const options = optionsOf(view)
  assert.ok(options, '点开知识库选择器后应当渲染出选项列表')

  const option = options.find((node) => node.textContent.trim() === label)
  assert.ok(option, `下拉里应当有「${label}」这个选项`)
  option.dispatchEvent(new MouseEvent('click', { bubbles: true }))
  await settle(view)
}

// aria-controls -> 该选择器自己的那一份选项列表（element-plus 给每个 select 生成的 id）。
function optionsOf(view) {
  const id = view.query('.el-select.w-48 input')?.getAttribute('aria-controls')
  const list = id ? globalThis.document.getElementById(id) : null
  return list ? [...list.querySelectorAll('.el-select-dropdown__item')] : null
}

// 打开重命名对话框并把名称改成 nextName，返回点击「保存」之前的水位线。
// currentName 是切换当前库之后应当被预填进去的名字（对应 `Knowledge.vue:436`）。
async function renameTo(view, nextName, currentName = 'KB2') {
  const button = view.buttonByText('重命名')
  assert.ok(button, '页面上应当有「重命名」按钮')
  assert.equal(
    button.disabled,
    false,
    '已选中知识库时「重命名」按钮应当可用（否则后面的断言只是空转）'
  )

  button.click()
  await until(view, () => dialogOpen(view))
  const input = view.query(NAME_INPUT)
  assert.ok(input, '点「重命名」后应当打开带名称输入框的知识库对话框')

  // `Knowledge.vue:436`：rename 模式的名称预填的是**当前知识库的名字**。
  // 少了这一行，用户点「重命名」看到的是空输入框，等于要重打一遍全名。
  assert.equal(input.value, currentName, '【:436】重命名对话框应当预填当前知识库的名称')

  const mark = calls.length
  // 走真实的 input 事件，让 el-input 的 v-model 与 el-form 的校验器都真的跑一遍：
  // 直接改 knowledgeBaseForm 会绕过 submitKnowledgeBaseDialog 开头的 validate()。
  input.value = nextName
  input.dispatchEvent(new Event('input', { bubbles: true }))
  await view.flush(3)
  return mark
}

async function submitRename(view) {
  const button = view.buttonByText('保存')
  assert.ok(button, '重命名对话框底部应当有「保存」按钮（对应 :370 的提交文案分支）')
  button.click()
  await settle(view)
}

async function close(view) {
  // ElMessage 的补丁挂在 element-plus 的模块对象上，跨用例共享：
  // 断言中途失败也必须还原，否则残留的补丁会污染后续用例记到的提示。
  view.elementPlusUnpatch()
  await view.unmount()
}

// ---------------------------------------------------------------------------
// a. 重命名成功：侧栏与当前选中库同步（验收 1 前半）
// ---------------------------------------------------------------------------

test('重命名成功：侧栏与当前选中库一起同步到新名字，且只刷新侧栏一次', async () => {
  const view = await mountKnowledge()

  try {
    assert.equal(currentBaseLabel(view), 'KB1（0）', '挂载后当前选中库应当是列表第一个')
    await switchBase(view, 'KB2（0）')
    assert.equal(
      currentBaseLabel(view),
      'KB2（0）',
      '点开下拉选中 KB2 之后当前选中库应当是 KB2（控制项：切库本身生效了）'
    )

    const mark = await renameTo(view, 'KB2-改名')
    await submitRename(view)

    // 先钉住「PUT 真的发出去了」：否则下面几条同步断言可能只是因为按钮没点动
    //（比如校验没过），是空转的绿灯。
    assert.deepEqual(
      requestsSince(mark),
      ['put /knowledge-bases/kb2', 'get /knowledge-bases'],
      '重命名成功后应当发出一次 PUT，并只刷新侧栏（rename 链没有第二次取数）'
    )
    assert.deepEqual(view.messages, [{ level: 'success', message: '知识库已重命名' }])

    // 承重①：store 真的被 upsert 了 —— 侧栏渲染出的是改名后的名字。
    assert.ok(
      view.text().includes('KB2-改名（0）'),
      '【:496】重命名成功后 store 应当被 upsert，侧栏渲染改名后的名字'
    )
    // 承重②：当前选中库仍然指向同一个库（:554 的 resolveKnowledgeBaseId 用重命名后的
    // 对象回填 preferred id）。少了它，选中会掉回列表第一个 —— 当前库此时是列表第二个，
    // 于是这个回落会以 `KB1（0）` 的形态露出来。
    assert.equal(
      currentBaseLabel(view),
      'KB2-改名（0）',
      '【:554】重命名成功后当前选中库应当仍然是被重命名的那个库，不得掉回列表第一个'
    )

    await until(view, () => !dialogOpen(view))
    assert.equal(dialogOpen(view), false, '重命名成功后对话框应当关闭')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// b. 承重：重命名成功但随后的取数失败（验收 1 后半，:542-556 的 upsert 回退）
// ---------------------------------------------------------------------------

test('重命名成功但侧栏刷新取数失败：upsert 回退生效，且不外泄成未处理拒绝', async () => {
  const rejections = []
  const onRejection = (error) => rejections.push(error)
  process.on('unhandledRejection', onRejection)

  const view = await mountKnowledge({ listErrorAfterRename: '连接中断' })

  try {
    await switchBase(view, 'KB2（0）')
    assert.equal(currentBaseLabel(view), 'KB2（0）', '切换后的当前选中库应当是 KB2（控制项）')

    const mark = await renameTo(view, 'KB2-改名')
    await submitRename(view)

    // 控制项：PUT 成功、刷新这次 GET 也确实发出去了（失败的是它的返回，不是没发）。
    assert.deepEqual(
      requestsSince(mark),
      ['put /knowledge-bases/kb2', 'get /knowledge-bases'],
      '取数失败不该让重命名请求本身被重发或被跳过'
    )

    // 承重①：`:548-552` 的 catch 分支把 preferred 对象 upsert 了回去。
    // 但它并非这条路径上唯一的救兵：`:496` 早在 `:502` 那次刷新之前就 upsert 过一次，
    // 两处互为冗余 —— 实测只删掉 catch 里这一处，本用例仍然全绿；两处一起拿掉才会红。
    // 所以下面两条断言钉的是用户可见的结果（当前选中库与侧栏都得跟着改名），
    // 而不是「一定是 `:550` 补回去的」。
    assert.equal(
      currentBaseLabel(view),
      'KB2-改名（0）',
      '【:550】取数失败时应当 upsert 回重命名后的对象，当前选中库不得停在旧名字上'
    )
    assert.ok(
      view.text().includes('KB2-改名（0）'),
      '【:550】取数失败时侧栏也必须显示改名后的名字'
    )

    // 承重②：`fetchKnowledgeBases()` 被包在 try/catch{} 里、catch 只 upsert 不重新抛出，
    // 所以这条路径上不得有未处理拒绝。反过来，若这里把 catch 改成重新抛出，
    // 下面这条断言与 node:test 自身的未处理拒绝检测都会红。
    assert.deepEqual(rejections, [], '取数失败不得外泄成未处理的 Promise 拒绝')

    // 刷新失败不改写操作的结论：重命名是既成事实，只提示成功那一条
    //（`Knowledge.vue:498-500` 就地注明这条链没有「刷新失败被当成操作失败」的可达路径）。
    assert.deepEqual(
      view.messages,
      [{ level: 'success', message: '知识库已重命名' }],
      '刷新取数失败不该被报成重命名失败'
    )
    assert.equal(dialogOpen(view), false, '重命名成功后对话框应当关闭，与刷新成不成功无关')
  } finally {
    process.off('unhandledRejection', onRejection)
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// c. 反向对照：重命名请求本身被拒 -> 不刷新、不提示成功、对话框留给用户重试
// ---------------------------------------------------------------------------

test('重命名请求被拒：只给一条错误提示、不刷新，对话框保持打开供重试', async () => {
  const view = await mountKnowledge({
    rename: () => {
      throw new Error('重命名接口不可用')
    },
  })

  try {
    await switchBase(view, 'KB2（0）')
    const mark = await renameTo(view, 'KB2-改名')
    await submitRename(view)

    // 成功分支（upsert + 关窗 + 刷新）整条都不该走：刷新会多打一次 GET。
    assert.deepEqual(
      requestsSince(mark),
      ['put /knowledge-bases/kb2'],
      '重命名失败时不应刷新侧栏'
    )
    assert.deepEqual(
      view.messages,
      [{ level: 'error', message: '重命名接口不可用' }],
      '重命名失败时应当只给一条错误提示（不多不少）'
    )
    assert.equal(dialogOpen(view), true, '重命名失败时对话框必须保持打开，用户才能改完重试')
    assert.equal(
      view.query(NAME_INPUT)?.value,
      'KB2-改名',
      '重命名失败时用户已经输入的名称不能丢'
    )
    assert.equal(currentBaseLabel(view), 'KB2（0）', '重命名失败时当前选中库不应变化')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// d. 边界：名字没改（同名）时短路，连请求都不发
// ---------------------------------------------------------------------------

test('名称没改：同名短路直接关窗，不发请求也不提示', async () => {
  const view = await mountKnowledge()

  try {
    await switchBase(view, 'KB2（0）')
    // 预填的名字就是当前名字，一个字符都不动直接提交。
    const mark = await renameTo(view, 'KB2')
    await submitRename(view)

    assert.deepEqual(requestsSince(mark), [], '【:459】名称未变化时不应发出重命名请求')
    assert.deepEqual(view.messages, [], '名称未变化是正常路径，不该有任何提示')
    await until(view, () => !dialogOpen(view))
    assert.equal(dialogOpen(view), false, '【:460】名称未变化时对话框应当直接关闭')
    assert.equal(currentBaseLabel(view), 'KB2（0）', '当前选中库不应变化')
  } finally {
    await close(view)
  }
})
