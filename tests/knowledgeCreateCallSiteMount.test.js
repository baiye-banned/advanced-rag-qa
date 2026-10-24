// Knowledge.vue 创建入口的**调用点胶水**挂载用例：真 mount 视图、点真按钮走完一次创建。
//
// 由来（issue #157）：创建成功之后的那次刷新原本是裸 await，留在 `submitKnowledgeBaseDialog`
// 自己的 try 里，而关窗那一行排在刷新**之后**。刷新一失败就跳去创建自己的 catch：
// 对话框停在打开态、弹一条归属不明的错误，而后端已经建了库、store 里也已经是新知识库 ——
// 两者互斥。用户照着这个还开着的对话框再点一次「创建」，拿到的是 400「知识库已存在」，
// 同一次操作给出两条互相矛盾的结论。删除面的同一处已在 #83 第 7 项收口、上传面在 #154 收口，
// 创建面当时缺席：**关窗的位置**与**刷新的归属**这两件事都只靠阅读，没有任何用例执行过。
//
// 与既有用例的分工（三份互不重叠）：
//   tests/knowledgeFeedback.test.js   注入替身直接测 refreshAfterCreate /
//                                     describeCreateRefreshFailure 的**纯函数本体**
//   tests/knowledgeViewWiring.test.js 静态读文件，证明「视图里接上了那个出口」
//   本文件                             两者之间：视图真的跑起来之后，刷新失败时到底弹了
//                                     什么、对话框最后是不是关着的
//
// 替身扎在**模块边界**上（api/request.js），被测代码照常真跑：knowledgeAPI、fileListRequest、
// pinia store 与视图全是真货，所以「刷新」表现为真实发出的 GET，对话框也是 Element Plus
// 自己渲染、自己关的（destroy-on-close 走完 leave 过渡，内容真的被销毁）。
//
// 断言一律落在 view.messages 的**完整数组**上，而不是「有没有出现某条文案」：
// #157 的缺陷形态之一就是「少了一条该有的、多了一条不该有的」，子集断言会从缝隙里漏过去。
//
// 本文件观测不到的一层：真实的 api/request.js 拦截器对刷新用的两个 GET 没有传 silent，
// 刷新失败时它自己也会弹一条通用错误提示（删除侧的刷新同理，见 #83；上传侧见 #154）。
// 挂载用例把整个 api/request.js 换成了替身，因此**拦截器那一层不在本文件的断言面内**，
// 这里钉的是视图自己发出的文案归属。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { createTestRouter } from './helpers/testRouter.js'
import { calls, resetRequestStub, respond } from './helpers/stubApiRequest.js'

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
}

const KB1 = { id: 'kb1', name: 'KB1' }
const NEW_BASE = { id: 'kb2', name: '新库' }
const FILES = [{ id: 'f1', name: 'a.pdf', size: 10, created_at: '2024-01-01T00:00:00Z' }]
const NAME_INPUT = 'input[placeholder="请输入知识库名称"]'

// 创建链的 await 层数是「点击 -> 表单校验 -> POST -> 成功提示 -> 关窗 -> 刷新知识库列表
// -> 刷新文件列表 -> 取数 -> 响应」，vueMount 默认的 3 轮不够深：浅了断言会看到「还没发生」
// 的假绿（尤其是「对话框必须已关」这类负向断言）。
//
// 光加轮数不够，还得给真实时间：flush 的每一轮都是 microtask + setTimeout(0)，整段在快机器上
// 几毫秒就跑完了，而对话框关闭走的是 Element Plus 的 leave 过渡，落地在 jsdom 的
// requestAnimationFrame 上（约 16ms），destroy-on-close 的销毁要等它走完。所以这里补一段
// 跨得过 rAF 的真实时间，与 knowledgeDeleteCallSiteMount.test.js 同一个理由。
const settle = async (view) => {
  await view.flush(8)
  await new Promise((resolve) => setTimeout(resolve, 40))
}

// 等一个可观测条件成立，超时就把当前状态原样交给后面的断言去报。
// 比定长 flush 可靠：条件什么时候成立由实现决定，不由轮数决定。
async function until(view, predicate, budgetMs = 1000) {
  const deadline = Date.now() + budgetMs
  for (;;) {
    if (predicate()) return true
    if (Date.now() > deadline) return false
    await view.flush(1)
    await new Promise((resolve) => setTimeout(resolve, 2))
  }
}

// 一次挂载 = 一次完整的视图启动（onMounted 会拉知识库列表与文件列表）。
//   create                 POST /knowledge-bases 的处理器；不传时一律成功返回 NEW_BASE
//   listErrorAfterCreate   非空时，**创建之后**那次 GET /knowledge 抛这个错误
//   hangListAfterCreate    true 时，**创建之后**那次 GET /knowledge 永不落地（在飞态）
//
// 失败/挂起必须只打在创建之后：挂载期那两次取数也要失败的话，视图连对话框都开不起来，
// 用例红的就不是被测的那条断言了。
async function mountKnowledge({ create, listErrorAfterCreate, hangListAfterCreate } = {}) {
  resetRequestStub()

  const world = { bases: [KB1], files: FILES }
  let created = false

  respond('get', (url) => {
    if (url === '/knowledge-bases') return world.bases
    if (url === '/knowledge') {
      if (listErrorAfterCreate && created) throw new Error(listErrorAfterCreate)
      if (hangListAfterCreate && created) return new Promise(() => {})
      return world.files
    }
    return {}
  })
  respond('post', (url, body) => {
    created = true
    // 服务端真的建了库：刷新拿到的列表里应当有它（下面两处断言都依赖这一点）。
    world.bases = [...world.bases, NEW_BASE]
    return create ? create(url, body) : NEW_BASE
  })

  const router = await createTestRouter()
  const view = await mountSfc('views/Knowledge.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  // 挂载期有两次 GET（知识库列表 + 文件列表）。等它们都落地再让调用方划水位线，
  // 否则迟到的挂载请求会混进「点击之后发了什么」里。
  await until(view, () => calls.filter((entry) => entry.method === 'get').length >= 2)
  return view
}

// 只看水位线之后的请求，形如 ['post /knowledge-bases', 'get /knowledge']。
function requestsSince(mark) {
  return calls.slice(mark).map((entry) => `${entry.method} ${entry.args[0]}`)
}

// 对话框是否还开着：判据取「对话框里的提交按钮还在不在」，也就是用户还能不能
// 照着它再点一次创建 —— 这正是 #157 里会把用户送进 400 的那个动作。
// destroy-on-close 走完 leave 过渡后内容被销毁，按钮随之消失。
function dialogOpen(view) {
  return Boolean(view.buttonByText('创建'))
}

async function close(view) {
  // ElMessage 的补丁挂在 element-plus 的模块对象上，跨用例共享：
  // 断言中途失败也必须还原，否则残留的补丁会污染后续用例记到的提示。
  view.elementPlusUnpatch()
  await view.unmount()
}

// 打开创建对话框并填好名称，返回点击「创建」之前的水位线。
async function openCreateDialog(view, name = '新库') {
  view.buttonByText('新建知识库').click()
  await until(view, () => Boolean(view.query(NAME_INPUT)))

  const input = view.query(NAME_INPUT)
  assert.ok(input, '应当渲染出创建对话框里的名称输入框')

  // 走真实的 input 事件，让 el-input 的 v-model 与 el-form 的校验器都真的跑一遍：
  // 直接改 knowledgeBaseForm 会绕过 submitKnowledgeBaseDialog 开头的 validate()。
  input.value = name
  input.dispatchEvent(new Event('input', { bubbles: true }))
  await view.nextTick()
  return calls.length
}

// ---------------------------------------------------------------------------
// a. 阳性对照：创建成功 + 刷新正常
// ---------------------------------------------------------------------------

test('创建成功且刷新正常：只有一条成功提示，对话框已关，两次刷新取数都真的发了', async () => {
  const view = await mountKnowledge()

  try {
    const mark = await openCreateDialog(view)
    view.buttonByText('创建').click()
    await settle(view)

    // 先钉住「创建请求真的发出去了」：否则下面两条断言可能只是因为按钮没点动
    // （比如校验没过），是空转的绿灯。
    assert.deepEqual(
      requestsSince(mark),
      ['post /knowledge-bases', 'get /knowledge-bases', 'get /knowledge'],
      '创建成功后应当发出一次 POST，并重新拉知识库列表与文件列表'
    )
    assert.deepEqual(view.messages, [{ level: 'success', message: '知识库已创建' }])

    await until(view, () => !dialogOpen(view))
    assert.equal(dialogOpen(view), false, '创建成功后对话框应当关闭')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// b. 承重：创建成功但随后的刷新失败（#157 本体）
// ---------------------------------------------------------------------------

test('创建成功但列表刷新失败：对话框已关，只多一条「知识库已创建，但列表刷新失败」', async () => {
  const view = await mountKnowledge({ listErrorAfterCreate: '连接中断' })

  try {
    const mark = await openCreateDialog(view)
    view.buttonByText('创建').click()
    await settle(view)

    // 创建那一步确实成了：POST 只发了一次，而且带着成功提示。
    assert.deepEqual(
      requestsSince(mark),
      ['post /knowledge-bases', 'get /knowledge-bases', 'get /knowledge'],
      '刷新失败不该让创建请求本身被重发或被跳过'
    )

    // 归属：先认下「知识库已创建」，再把刷新没跟上单独说清。
    // 修复前这里是 [success, error「连接中断」] —— 刷新的原始错误被创建自己的 catch 弹出，
    // 文案里看不出创建已经成功。
    assert.deepEqual(
      view.messages,
      [
        { level: 'success', message: '知识库已创建' },
        { level: 'error', message: '知识库已创建，但列表刷新失败：连接中断' },
      ],
      '创建成功的结论必须保留，刷新失败另起一条并说明归属'
    )

    // 修复前对话框停在打开态，用户照着它再点一次「创建」就是 400「知识库已存在」。
    await until(view, () => !dialogOpen(view))
    assert.equal(
      dialogOpen(view),
      false,
      '创建成功那一刻对话框就该关掉，不该由随后的刷新决定它留在打开态'
    )
    // 对话框关了，里面输入的名称也随之销毁 —— 这正是修复前「重打一遍」的由来，
    // 反过来也证明这个探针看的是真内容，不是恒为假的空断言。
    assert.equal(view.query(NAME_INPUT), null, '关窗后对话框内容应当已被销毁')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// c. 承重（时序面）：刷新还在飞的时候，对话框就已经关了
// ---------------------------------------------------------------------------

// 上一节钉的是「刷新失败之后对话框是关的」，这一节钉的是它**什么时候**关的：
// 只要关窗还排在刷新后面，这两件事就随时可以被一次慢刷新重新绑在一起 ——
// 刷新迟迟不落地时用户盯着一个已经建好库的对话框，可以再点一次创建。
// 把刷新挂成永不落地，判据就退化成纯粹的顺序问题：不等刷新，也该是关的。
test('创建成功但刷新还没落地：对话框不等刷新就已经关了', async () => {
  const view = await mountKnowledge({ hangListAfterCreate: true })

  try {
    const mark = await openCreateDialog(view)
    view.buttonByText('创建').click()

    // 只等「两次刷新取数都已经发出去」，不等它们落地 —— 此刻文件列表那次仍在在飞态。
    await until(view, () => requestsSince(mark).includes('get /knowledge'))
    assert.deepEqual(
      requestsSince(mark),
      ['post /knowledge-bases', 'get /knowledge-bases', 'get /knowledge'],
      '刷新请求确实发出去了，只是还没有返回'
    )

    await until(view, () => !dialogOpen(view))
    assert.equal(
      dialogOpen(view),
      false,
      '刷新还挂着的当下对话框就该是关的：关窗排在刷新之前，它的去留不取决于刷新'
    )
    // 刷新没落地就不该有「刷新失败」的说法：此刻只该有创建成功那一条。
    assert.deepEqual(
      view.messages,
      [{ level: 'success', message: '知识库已创建' }],
      '刷新还在飞，不能预先报它有结果'
    )
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// d. 反向对照：创建请求本身被拒 -> 错误路径不失去提示，对话框留给用户重试
// ---------------------------------------------------------------------------

test('创建请求本身被拒：只有一条错误提示、不刷新，对话框保持打开供重试', async () => {
  const view = await mountKnowledge({
    create: () => {
      throw new Error('创建接口不可用')
    },
  })

  try {
    const mark = await openCreateDialog(view)
    view.buttonByText('创建').click()
    await settle(view)

    // 关窗上移只该改「刷新失败」那条路径：创建请求自己失败仍然是失败，
    // 既不能提示成功，也不能刷新（刷新会打两次 GET）。
    assert.deepEqual(requestsSince(mark), ['post /knowledge-bases'], '创建失败时不应刷新')
    assert.deepEqual(
      view.messages,
      [{ level: 'error', message: '创建接口不可用' }],
      '创建失败时应当只给一条错误提示（不多不少）'
    )
    assert.equal(dialogOpen(view), true, '创建失败时对话框必须保持打开，用户才能改完重试')
    assert.equal(
      view.query(NAME_INPUT)?.value,
      '新库',
      '创建失败时用户已经输入的名称不能丢'
    )
  } finally {
    await close(view)
  }
})
