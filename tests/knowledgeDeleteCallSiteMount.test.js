// Knowledge.vue 三个删除入口的**调用点胶水**挂载用例：真 mount 视图、点真按钮。
//
// docs/MAINTENANCE_GOAL_CLOSURE.md 的 Follow-Up 第 5 条自陈：视图没有挂载用例，
// 于是没被覆盖的正是调用点胶水本身 —— `status !== DELETE_SUCCEEDED` 那句短路，
// 以及紧跟其后的那次刷新。本文件补的就是这块：短路有没有真的拦住刷新，
// 刷新又有没有真的发出去。渲染细节不是本文件的断言面。
//
// 与既有用例的分工（三份互不重叠）：
//   tests/knowledgeFeedback.test.js   注入替身直接测 runConfirmedDelete /
//                                     refreshAfterDelete 的**纯函数本体**（三态返回值）
//   tests/knowledgeViewWiring.test.js 静态读文件，证明「视图里写了这几行」
//   本文件                             两者之间：视图真的跑起来之后，那几行有没有生效
//
// 替身只扎在**一个**模块边界上（issue #180 验收 4 之后的形态）：
//   api/request.js -> stubApiRequest.js   只换 HTTP 出口；knowledgeAPI、
//                                         fileListRequest、pinia store、utils/confirm.js
//                                         与视图全是真货，所以「刷新」表现为真实发出的 GET
//
// **确认框不再打桩**：本文件此前把 `utils/confirm.js` 换成 tests/helpers/stubConfirm.js，
// 于是「用户取消 -> 静默返回」这条分支从未在真实 `ElMessageBox` 下执行过，而
// `runConfirmedDelete` 正是靠 `isConfirmCancellation` 识别真实弹窗那个 'cancel' 字符串的。
// 打桩理由（stubConfirm.js 头注释记的 focus-trap 缺 `HTMLInputElement` 全局）**已经过期**：
// vueMount.js 现在会把 jsdom 上的元素类逐个搬到 Node 全局，真开弹窗、点取消、点删除三条
// 路径全部可跑。本文件因此改成真开弹窗、真点按钮 —— 归属断言（哪些请求、哪些提示）
// 与弹窗实现一并被执行。
//
// 一处必须写明的取舍：**整个文件只能有一套模块替换表**。Node 的 ESM 缓存按 URL 记模块，
// 同一个 SFC 在第二次挂载时不会重新解析依赖，所以「一部分用例用真实确认框、另一部分用替身」
// 在同一个文件里做不到 —— 先挂载的那一套会被后来者继承。要么全真、要么全替身，
// 这里选了全真。取消与确认两条出口分别由 a 节与 b/c 节覆盖。
//
// 断言因此落在「发了哪些请求」「弹了哪些提示」与「真实确认框的哪个按钮被点」上，
// 而不是渲染兜底。

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
// 单条删除的用例只放一个文件：行内「删除」按钮按文本取的是第一个，
// 列表顺序又随默认排序字段走，放两个会让「删的是哪一行」变成排序的间接后果。
const ONE_FILE = [{ id: 'f1', name: 'a.pdf', size: 10, created_at: '2024-01-01T00:00:00Z' }]
const TWO_FILES = [
  ...ONE_FILE,
  { id: 'f2', name: 'b.md', size: 20, created_at: '2024-01-02T00:00:00Z' },
]

// 服务端状态的替身。删除处理器可以就地改它，刷新拿到的就是「删完之后」的世界。
function createWorld({ bases = [KB1, KB2], files = ONE_FILE } = {}) {
  return { bases, files }
}

// 一次挂载 = 一次完整的视图启动（onMounted 会拉知识库列表与文件列表）。
// remove 是 HTTP 边界的删除处理器；不传时删除一律成功。
async function mountKnowledge({ world = createWorld(), remove } = {}) {
  resetRequestStub()
  respond('get', (url) => {
    if (url === '/knowledge-bases') return world.bases
    if (url === '/knowledge') return world.files
    return {}
  })
  if (remove) respond('delete', remove)

  const router = await createTestRouter()
  const view = await mountSfc('views/Knowledge.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  // 挂载期有两次 GET（知识库列表 + 文件列表）。等它们都落地再让调用方划水位线，
  // 否则迟到的挂载请求会混进「点击之后发了什么」里。
  await until(view, () => calls.filter((entry) => entry.method === 'get').length >= 2)
  return view
}

// 删除链路的 await 层数是「确认 -> 执行删除 -> 刷新 -> 取数 -> 响应」，
// vueMount 默认的 3 轮不够深：浅了断言会看到「还没发生」的假绿（尤其是负向断言）。
//
// 光加轮数不够，还得给真实时间：flush 的每一轮都是 microtask + setTimeout(0)，
// 整段在快机器上几毫秒就跑完了，而 jsdom 的 requestAnimationFrame 约 16ms 才触发，
// 被 rAF 推迟的那次更新（el-table 的勾选状态就在这条路径上）会整个错过。
// 这个缺口的表现是**机器越快越容易红**：本地（慢）绿、CI（快）红，
// 用 CPU 满载把本地压慢反而复现不出来。所以这里补一段跨得过 rAF 的真实时间。
const settle = async (view) => {
  await view.flush(8)
  await new Promise((resolve) => setTimeout(resolve, 40))
}

// 等一个可观测条件成立，超时就把当前状态原样交给后面的断言去报。
// 比定长 flush 可靠：条件什么时候成立由实现决定，不由轮数决定。
async function until(view, predicate, budgetMs = 2000) {
  const deadline = Date.now() + budgetMs
  for (;;) {
    if (predicate()) return true
    if (Date.now() > deadline) return false
    await view.flush(1)
    await new Promise((resolve) => setTimeout(resolve, 2))
  }
}

// 只看水位线之后的请求，形如 ['delete /knowledge/f1', 'get /knowledge-bases']。
// 挂载本身会打两次 GET，不划水位线就分不清「刷新」与「启动」。
function requestsSince(mark) {
  return calls.slice(mark).map((entry) => `${entry.method} ${entry.args[0]}`)
}

// ---------------------------------------------------------------------------
// 真实确认框的访问面
// ---------------------------------------------------------------------------

// 弹窗由 `utils/confirm.js` 的 `appendTo: 'body'` 交给 element-plus 自己 append，
// 而**视图的 host 也在 body 里** —— 所以只能在 `.el-message-box` 范围内找按钮：
// 页面上行内那排删除按钮的文案也叫「删除」，在 body 上全局找会打错目标。
function messageBoxes() {
  return [...globalThis.document.querySelectorAll('.el-message-box')]
}

function boxButton(box, label) {
  return [...box.querySelectorAll('button')].find((node) => node.textContent.trim() === label)
}

async function untilMessageBox(view, count = 1) {
  return until(view, () => messageBoxes().length >= count)
}

async function untilMessageBoxGone(view) {
  return until(view, () => messageBoxes().length === 0)
}

// 开一次真实确认框并点其中一个按钮。
// 打开与关闭都要等：ElMessageBox 是函数式调用（先 append 再挂载组件），关闭还带 leave 过渡。
// 返回弹窗元素，便于调用方对文案再作断言。
async function answerConfirm(view, label) {
  assert.ok(await untilMessageBox(view), '点删除入口后应当开出真实的 ElMessageBox')
  // 上一个弹窗若还在离场过渡里，body 里可能同时存在两个：取最后一个（本次开的那个）。
  const box = messageBoxes().at(-1)
  const button = boxButton(box, label)
  assert.ok(button, `真实确认框里应当有「${label}」按钮`)
  button.click()
  assert.ok(await untilMessageBoxGone(view), '点完按钮后真实确认框应当关闭')
  return box
}

async function close(view) {
  // ElMessage 的补丁挂在 element-plus 的模块对象上，跨用例共享：
  // 断言中途失败也必须还原，否则残留的补丁会污染后续用例记到的提示。
  view.elementPlusUnpatch()
  await view.unmount()
}

// 三个入口共用的驱动：label 是按钮文案，selectAll 表示要不要先勾选资料。
// 批量删除按钮在没勾选时是禁用的，勾选走表头的全选框（applied 后按钮才可用）。
const ENTRIES = [
  { label: '删除知识库', target: 'delete /knowledge-bases/kb1' },
  { label: '删除', target: 'delete /knowledge/f1' },
  { label: '删除选中', target: 'delete /knowledge/f1', selectAll: true },
]

// answer 是真实确认框上要点的按钮文案：'取消' 走放弃路径，'删除' 放行。
async function hitEntry(view, entry, answer) {
  if (entry.selectAll) {
    // 表头全选框存在 = 文件列表已经从空态切到 el-table。点它之前先确认拿到的是
    // 真元素，否则后面的失败会伪装成「删除请求数为 0」，看不出是勾选没生效。
    const box = view.queryAll('.el-checkbox__input')[0]
    assert.ok(box, '应当渲染出表头全选框')
    box.click()

    // 勾选要经 el-table 的 selection-change 回流到 selectedFiles 才会让按钮可用。
    // 这一步单独断言：一旦它没生效，报的是「选中没生效」而不是「没有删除请求」——
    // 后者会把人往短路/刷新的方向带偏（CI 上第一版就是这么红的）。
    await until(view, () => view.buttonByText('删除选中').disabled === false)
    assert.equal(
      view.buttonByText('删除选中').disabled,
      false,
      '勾选全部资料后，批量删除按钮应当可用（否则后续「0 个删除请求」只是勾选没生效）'
    )
  }
  const mark = calls.length
  view.buttonByText(entry.label).click()
  // 真实确认框先开出来，再点 answer 指定的那个按钮。
  const box = await answerConfirm(view, answer)
  await settle(view)
  return { mark, box }
}

// ---------------------------------------------------------------------------
// a. 取消确认：三个入口都静默返回（真实确认框点「取消」）
// ---------------------------------------------------------------------------

test('取消确认：三个删除入口都静默返回，不删任何东西、不刷新、不提示', async () => {
  for (const entry of ENTRIES) {
    const view = await mountKnowledge({
      world: createWorld({ files: entry.selectAll ? TWO_FILES : ONE_FILE }),
      remove: () => ({ fallback_knowledge_base_id: 'kb2' }),
    })

    try {
      const { mark } = await hitEntry(view, entry, '取消')

      // 「确认框真的被问过一次」由 hitEntry 里的 untilMessageBox + 按钮点名断言保证：
      // 弹窗没开出来、或里面没有那个按钮，那里就先红了。缺了这层，下面几条静默断言
      // 可能只是因为按钮压根没点动（比如禁用态），是空转的绿灯。
      assert.deepEqual(requestsSince(mark), [], `${entry.label}：取消后不应发出任何请求`)
      assert.deepEqual(view.messages, [], `${entry.label}：取消是正常路径，不该有任何提示`)
      // 取消不该改动视图状态：知识库列表里那个知识库还在。
      assert.ok(view.text().includes('KB1'), `${entry.label}：取消后视图状态不应变化`)
    } finally {
      await close(view)
    }
  }
})

// ---------------------------------------------------------------------------
// b. 接口拒绝 -> status !== DELETE_SUCCEEDED -> 不进刷新分支（承重：短路）
// ---------------------------------------------------------------------------

test('接口拒绝：两个单条删除入口都不进刷新分支，错误只提示一次', async () => {
  for (const entry of ENTRIES.filter((item) => !item.selectAll)) {
    const view = await mountKnowledge({
      world: createWorld({ files: ONE_FILE }),
      remove: () => {
        throw new Error('删除接口不可用')
      },
    })

    try {
      const { mark } = await hitEntry(view, entry, '删除')

      // 短路要否掉的是**刷新**，不是删除本身：先把「删除已经真的发出去了」
      // 钉死，断言才落在短路那一行上，而不是退化成一堆“什么都没发生”。
      assert.deepEqual(
        requestsSince(mark),
        [entry.target],
        `${entry.label}：确认放行后应当恰好发出一次删除请求`
      )
      // 短路承重：失败态既不能提示成功，也不能走到刷新（刷新会打 GET）。
      assert.deepEqual(
        view.messages,
        [{ level: 'error', message: '删除接口不可用' }],
        `${entry.label}：失败时应当只给一条错误提示（不多不少）`
      )
    } finally {
      await close(view)
    }
  }
})

test('接口拒绝：视图不假装删掉了，资料与选中状态都原样保留', async () => {
  // 单条删除失败：文件仍在列表里，没有被乐观移除。
  const single = await mountKnowledge({
    remove: () => {
      throw new Error('删除接口不可用')
    },
  })
  try {
    await hitEntry(single, ENTRIES[1], '删除')
    assert.ok(single.text().includes('a.pdf'), '删除失败后资料不应从列表消失')
  } finally {
    await close(single)
  }

  // 批量删除失败：allSettled 把所有拒绝收敛成 succeeded=0 的结果，
  // 此时服务端没有任何变化，选中必须保留，用户才能直接重试。
  const batch = await mountKnowledge({
    world: createWorld({ files: TWO_FILES }),
    remove: () => {
      throw new Error('删除接口不可用')
    },
  })
  try {
    const { mark } = await hitEntry(batch, ENTRIES[2], '删除')

    assert.deepEqual(
      requestsSince(mark).sort(),
      ['delete /knowledge/f1', 'delete /knowledge/f2'],
      '批量删除应当对每个选中项各发一次删除请求'
    )
    assert.deepEqual(
      batch.messages,
      [{ level: 'error', message: '已删除 0 个资料，2 个删除失败' }],
      '整批失败时应当如实报 0 删除、2 失败，不能谎报成功'
    )
    // 承重：hasDeletedAnyFile 为假时必须原地返回。少了它，选中的勾会被清掉、
    // 还要多打两次刷新，而用户什么都没删成。
    assert.equal(
      batch.buttonByText('删除选中').disabled,
      false,
      '整批失败时选中状态必须保留，用户可以直接重试'
    )
    assert.deepEqual(requestsSince(mark).filter((item) => item.startsWith('get')), [], '整批失败时不应刷新')
  } finally {
    await close(batch)
  }
})

// ---------------------------------------------------------------------------
// c. 删除成功 -> 刷新真的发出去（承重：refresh）
// ---------------------------------------------------------------------------

test('删除成功：文件删除与批量删除都会刷新列表', async () => {
  const expected = {
    删除: {
      files: ONE_FILE,
      remove: () => ({}),
      requests: ['delete /knowledge/f1', 'get /knowledge-bases', 'get /knowledge'],
      message: { level: 'success', message: '删除成功' },
    },
    删除选中: {
      files: TWO_FILES,
      remove: () => ({}),
      requests: [
        'delete /knowledge/f1',
        'delete /knowledge/f2',
        'get /knowledge-bases',
        'get /knowledge',
      ],
      message: { level: 'success', message: '已删除选中资料' },
    },
  }

  for (const entry of ENTRIES.filter((item) => item.label !== '删除知识库')) {
    const spec = expected[entry.label]
    const view = await mountKnowledge({ world: createWorld({ files: spec.files }), remove: spec.remove })

    try {
      const { mark } = await hitEntry(view, entry, '删除')

      // 刷新承重：删完之后必须重新拉一次知识库列表与文件列表。
      // 批量的删除请求由 allSettled 并发发出，落地顺序不作断言（顺序无关语义）；
      // 「先侧栏后列表」的顺序在删知识库那条用例里逐字断言。
      assert.deepEqual(
        requestsSince(mark).sort(),
        [...spec.requests].sort(),
        `${entry.label}：删除成功后应当发出这些请求`
      )
      assert.deepEqual(view.messages, [spec.message], `${entry.label}：成功提示`)
    } finally {
      await close(view)
    }
  }
})

test('删除知识库成功：刷新后当前选中落到响应给的 fallback 知识库', async () => {
  const world = createWorld()
  const view = await mountKnowledge({
    world,
    remove: () => {
      // 服务端真的删掉了 kb1，刷新拿到的列表里不该再有它。
      world.bases = [KB2]
      return { fallback_knowledge_base_id: 'kb2' }
    },
  })

  try {
    const { mark } = await hitEntry(view, ENTRIES[0], '删除')

    assert.deepEqual(requestsSince(mark), [
      'delete /knowledge-bases/kb1',
      'get /knowledge-bases',
      'get /knowledge',
    ])
    assert.deepEqual(view.messages, [{ level: 'success', message: '知识库已删除' }])
    // 选中要落到删除响应给的 fallback 上。它来自**已经成功**的删除响应，
    // 不依赖这次刷新；去掉那段赋值，刷新侧栏一失败选中就停在刚被删掉的 kb1 上，
    // 后续上传/删除都会以 404 收场。
    assert.match(view.text(), /KB2（0）/, '删除后选中应当落到 fallback 知识库上')
    assert.doesNotMatch(view.text(), /KB1/, '已被删除的知识库不应再出现在视图里')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// d. 真实确认框本身：文案与两个按钮都由视图传下去的参数决定
// ---------------------------------------------------------------------------

// 前几节用的是真实弹窗，但断言都落在「点完之后发生了什么」上。这一节反过来钉弹窗自己：
// 视图传下去的文案与按钮文案真的到了 DOM 上 —— 否则「点了取消」可能只是点到一个
// 文案对不上的按钮，用户看到的确认框和被测的那句话不是同一句。
test('真实确认框：文案与「取消 / 删除」两个按钮都与调用点传入的一致', async () => {
  const view = await mountKnowledge({ world: createWorld() })

  try {
    view.buttonByText('删除知识库').click()
    assert.ok(await untilMessageBox(view), '点「删除知识库」应当开出真实的 ElMessageBox')

    const box = messageBoxes().at(-1)
    assert.match(
      box.textContent,
      /确定删除知识库「KB1」吗？该知识库下的资料会一并删除，已有对话将切换到其他知识库。/,
      '【:589】确认框里应当是视图传进去的那句文案'
    )
    assert.ok(boxButton(box, '取消'), '真实确认框里应当有「取消」按钮')
    assert.ok(boxButton(box, '删除'), '真实确认框里应当有「删除」按钮（对应 confirmButtonText）')

    // 收尾：点取消关掉它，别把弹窗留给后面的用例。
    boxButton(box, '取消').click()
    assert.ok(await untilMessageBoxGone(view), '点「取消」后弹窗应当关闭')
    assert.deepEqual(view.messages, [], '取消不该产生任何提示')
  } finally {
    await close(view)
  }
})
