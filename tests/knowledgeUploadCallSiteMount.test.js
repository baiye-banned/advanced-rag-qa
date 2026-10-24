// Knowledge.vue 上传入口的**调用点胶水**挂载用例：真 mount 视图、真的走一次文件选择。
//
// 由来（issue #154）：上传成功后的那次刷新原本留在上传自己的 try 里，刷新一失败就被上传的
// catch 接走，而 `failedIndex < 0` 在上传成功时恒成立，于是逐字弹「上传失败，请稍后重试」——
// 文件其实已经入库，列表只是没跟上。用户据此重传会**真的再入一份**（后端对文件名没有唯一
// 约束，重复入库还附带重复分块与向量化开销）。删除面的同一处已在 #83 第 7 项收口，
// 上传面当时缺席：knowledgeViewWiring.test.js 把上传排除在计数之外，理由是「有自己的
// try/catch」，而那只保证不逃逸成未捕获拒绝，保证不了文案归属。
//
// 与既有用例的分工（三份互不重叠）：
//   tests/knowledgeFeedback.test.js   注入替身直接测 refreshAfterUpload /
//                                     describeUploadRefreshFailure 的**纯函数本体**
//   tests/knowledgeViewWiring.test.js 静态读文件，证明「视图里接上了那个出口」
//   本文件                             两者之间：视图真的跑起来之后，刷新失败时到底弹了
//                                     什么、上传成功的事实有没有被改写
//
// 替身扎在**模块边界**上（api/request.js），被测代码照常真跑：knowledgeAPI、fileListRequest、
// pinia store 与视图全是真货，所以「刷新」表现为真实发出的 GET。
//
// 断言一律落在 view.messages 的**完整数组**上，而不是「有没有出现某条文案」：
// #154 的缺陷形态就是「多了一条」，子集断言会从缝隙里把它漏过去。
//
// 本文件观测不到的一层：真实的 api/request.js 拦截器对刷新用的两个 GET 没有传 silent，
// 刷新失败时它自己也会弹一条通用错误提示（删除侧的刷新同理，见 #83）。挂载用例把整个
// api/request.js 换成了替身，因此**拦截器那一层不在本文件的断言面内**，这里钉的是视图
// 自己发出的文案归属。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { createTestRouter } from './helpers/testRouter.js'
import { calls, resetRequestStub, respond } from './helpers/stubApiRequest.js'

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
}

const KB1 = { id: 'kb1', name: 'KB1' }
const OLD_FILE = { id: 'f1', name: 'old.pdf', size: 10, created_at: '2024-01-01T00:00:00Z' }
const UPLOADED = { id: 'f2', name: 'a.pdf', size: 10, created_at: '2024-01-02T00:00:00Z' }

// 上传链的 await 层数是「change -> handleUpload -> 顺序上传 -> 上传请求 -> 成功提示 ->
// 刷新 -> 取数 -> 响应」，vueMount 默认的 3 轮不够深：浅了断言会看到「还没发生」的假绿
// （尤其是「不得出现某条文案」这类负向断言）。
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

// 只看水位线之后的请求，形如 ['post /knowledge/upload', 'get /knowledge-bases']。
// 挂载本身会打两次 GET，不划水位线就分不清「刷新」与「启动」。
function requestsSince(mark) {
  return calls.slice(mark).map((entry) => `${entry.method} ${entry.args[0]}`)
}

// 一次挂载 = 一次完整的视图启动（onMounted 会拉知识库列表与文件列表）。
//
// refreshFailsOn 收 URL、返回错误或 null：
//   null             刷新一律正常（阳性对照与正常结局）
//   () => ({...})    刷新期每个 GET 都抛
//   (url) => ...     按 URL 挑着抛（用于「侧栏成功、文件列表失败」）
//
// 关键在于**只在刷新期生效**（inRefresh 开关），而不是「第几个 GET 开始失败」：
// 挂载期的两次 GET 必须成功，否则视图根本起不来，后面测的就不是上传链了。
// recon 探针第一版正是在这里栽的——条件写反，阳性对照被驱动代码自己喂成了失败，
// 假红差点被当成结论，所以这里用一个显式开关而不是计数。
async function mountKnowledge({ refreshFailsOn = null, upload } = {}) {
  resetRequestStub()

  const world = { bases: [KB1], files: [OLD_FILE] }
  let inRefresh = false

  respond('get', (url) => {
    if (inRefresh && refreshFailsOn) {
      const error = refreshFailsOn(url)
      if (error) throw error
    }
    if (url === '/knowledge-bases') return world.bases
    if (url === '/knowledge') return world.files
    return {}
  })
  respond('post', upload || (() => UPLOADED))

  const router = await createTestRouter()
  const view = await mountSfc('views/Knowledge.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  await until(view, () => calls.filter((entry) => entry.method === 'get').length >= 2)

  inRefresh = true
  return { view, mark: calls.length }
}

// 走真入口：隐藏的 <input type="file" multiple> 拿到文件后派发 change，
// handleUploadInputChange -> handleUpload 整条链真跑。
// jsdom 不允许直接给 input.files 赋值，用 defineProperty 造出 FileList 的位置。
function pickFiles(view, names) {
  const input = view.query('input[type=file]')
  assert.ok(input, '应当渲染出隐藏的文件选择框')
  const files = names.map((name) => new File(['hello'], name, { type: 'application/pdf' }))
  Object.defineProperty(input, 'files', { value: files, configurable: true })
  input.dispatchEvent(new Event('change'))
}

async function uploadOnce({ names = ['a.pdf'], ...options } = {}) {
  const { view, mark } = await mountKnowledge(options)
  pickFiles(view, names)
  await settle(view)
  return { view, mark }
}

// 等提示条数到位再断言：链路的最后一步是刷新，提示在它之后才发出。
// 先等到位再 deepEqual，语义是「完整结局」而不是「此刻恰好」——
// 也让失败时报出来的是真正的文案差异，而不是「还没弹出来」的时序假红。
async function expectMessages(view, expected) {
  await until(view, () => view.messages.length >= expected.length)
  await settle(view)
  assert.deepEqual(view.messages, expected)
}

async function close(view) {
  // ElMessage 的补丁挂在 element-plus 的模块对象上，跨用例共享：
  // 断言中途失败也必须还原，否则残留的补丁会污染后续用例记到的提示。
  view.elementPlusUnpatch()
  await view.unmount()
}

// ---------------------------------------------------------------------------
// 阳性对照：正常结局。它同时证明本文件的驱动方式真的驱动到了上传链——
// 少了这一条，下面所有「不得出现上传失败」的断言都可能只是没走到上传。
// ---------------------------------------------------------------------------

test('阳性对照：刷新正常时只有一条成功提示，上传与刷新都真的发出', async () => {
  const { view, mark } = await uploadOnce()

  try {
    // 上传请求必须真的发出，否则后面测的不是上传链。
    assert.deepEqual(
      requestsSince(mark),
      ['post /knowledge/upload', 'get /knowledge-bases', 'get /knowledge'],
      '上传成功后应当刷新知识库列表与文件列表'
    )
    await expectMessages(view, [{ level: 'success', message: '上传成功' }])
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// 承重用例：刷新失败不得被改写成「上传失败」（第 154 项）
// ---------------------------------------------------------------------------

test('刷新失败：保留上传成功的结论，只加一条「上传成功，但列表刷新失败」', async () => {
  // 抛一个既无 message 也无 detail 的错误：修复前 getApiErrorMessage 回落到
  // 「上传失败，请稍后重试」，用户逐字看到的结论与事实相反。
  const { view, mark } = await uploadOnce({
    refreshFailsOn: () => ({ code: 'ECONNABORTED' }),
  })

  try {
    // 先把「文件真的传上去了」钉死：这是本用例的前提，不是结论。
    assert.deepEqual(
      requestsSince(mark).filter((request) => request.startsWith('post')),
      ['post /knowledge/upload'],
      '上传请求必须已经发出并成功（否则不是本缺陷）'
    )
    // 刷新也真的发起过（第一个 GET 失败后 refreshKnowledgeBaseAndFiles 不再走第二步）。
    assert.deepEqual(
      requestsSince(mark).filter((request) => request.startsWith('get')),
      ['get /knowledge-bases'],
      '刷新应当真的发起过'
    )

    await expectMessages(view, [
      { level: 'success', message: '上传成功' },
      { level: 'error', message: '上传成功，但列表刷新失败：请手动刷新页面' },
    ])
    // 承重：三条文案里不得有一条把这次操作说成上传失败——重传会再入一份。
    assert.ok(
      view.messages.every((message) => !message.message.includes('上传失败')),
      '刷新失败不得被报成上传失败：文件已经入库，用户据此重传会再入一份'
    )
  } finally {
    await close(view)
  }
})

test('刷新失败（侧栏成功、文件列表失败）：同样不改写上传成功的结论', async () => {
  // 刷新是「先取知识库列表、再取文件列表」两段，这里让第二段失败：
  // 侧栏已经拿到新数据、文件列表还停在旧数据，文案归属不变。
  const { view, mark } = await uploadOnce({
    refreshFailsOn: (url) => (url === '/knowledge' ? new Error('网络连接已断开') : null),
  })

  try {
    assert.deepEqual(
      requestsSince(mark).filter((request) => request.startsWith('get')),
      ['get /knowledge-bases', 'get /knowledge'],
      '两段刷新都应当发起'
    )
    await expectMessages(view, [
      { level: 'success', message: '上传成功' },
      { level: 'error', message: '上传成功，但列表刷新失败：网络连接已断开' },
    ])
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// 反向用例：修复不得把**真的**上传失败也说成刷新失败。
// 少了这一条，一个「刷新失败吞掉上传错误」的实现同样能让上面两条变绿。
// ---------------------------------------------------------------------------

test('真上传失败：仍逐字报上传失败，且不刷新', async () => {
  const { view, mark } = await uploadOnce({
    upload: () => {
      throw new Error('上传接口不可用')
    },
  })

  try {
    // 上传本身失败时没有「已经入库」这回事，不得刷新、不得出现刷新失败文案。
    assert.deepEqual(
      requestsSince(mark),
      ['post /knowledge/upload'],
      '上传失败后不应刷新（列表没有变化）'
    )
    await expectMessages(view, [
      {
        level: 'error',
        message: '「a.pdf」上传失败：上传接口不可用；同批其余文件均已上传',
      },
    ])
    assert.ok(
      view.messages.every((message) => !message.message.includes('刷新失败')),
      '上传自己的失败不得被说成刷新失败'
    )
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// 结局的其余两面：批量文案，以及刷新失败后的状态复位
// ---------------------------------------------------------------------------

test('批量上传成功：提示报出批量大小，刷新照常发起', async () => {
  const { view, mark } = await uploadOnce({ names: ['a.pdf', 'b.md'] })

  try {
    assert.deepEqual(
      requestsSince(mark).filter((request) => request.startsWith('post')),
      ['post /knowledge/upload', 'post /knowledge/upload'],
      '两个文件应当各发一次上传请求'
    )
    await expectMessages(view, [{ level: 'success', message: '已上传 2 个文件' }])
  } finally {
    await close(view)
  }
})

test('刷新失败：上传态照常复位，不因刷新失败卡在「上传中」', async () => {
  // 复位的 finally 在刷新之前，所以刷新成功与否都得复位；这条防的是「把刷新挪出去时
  // 顺手把 finally 也挪走」——那样按钮会一直停在「上传中 N%」。
  const { view } = await uploadOnce({
    refreshFailsOn: () => ({ code: 'ECONNABORTED' }),
  })

  try {
    const button = view.buttonByText('上传文件')
    assert.ok(button, '刷新失败后按钮文案应当复位成「上传文件」')
    assert.equal(button.disabled, false, '复位后应当可以继续上传')
  } finally {
    await close(view)
  }
})
