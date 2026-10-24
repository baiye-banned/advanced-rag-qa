// Knowledge.vue 从地址恢复详情预览的挂载用例（issue #255）。
//
// 地址契约：/knowledge?file=<正整数文件 id>
// Chat 的参考资料卡片跳过来时只给这个地址，不再依赖跨页的内存状态，因此
// 「刷新 / 直接访问」与「从会话页点过来」走的是同一条恢复路径。本文件把视图真挂起来，
// 断言这条路径上的四件事：
//   1. 目标不在列表里 -> 取元数据再取正文，弹窗显示的是这个文件；
//   2. 目标已在列表里 -> 直接用列表行，不再重复请求元数据；
//   3. 文件不存在（404 / 空响应）-> 友好提示并清掉 file query，其余 query 保留，
//      且拒绝被视图接住，不逃逸成未处理拒绝；
//   4. 快速切换目标 -> 迟到的旧响应不得打开错误的文件。
//
// 与既有用例的分工：
//   tests/knowledgeViewWiring.test.js  静态读源码，钉「接线写在文件里」
//   tests/detailPreview.test.js        纯函数本体（时效守卫、错误收口）
//   本文件                             真 mount 视图 + 真路由，验证 query -> 预览的胶水
//
// 替身只扎在 api/request.js 一个边界上；knowledgeAPI、detailPreview、pinia store、
// 视图本身全是真货，所以「打开详情」表现为真实发出的 GET。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { createTestRouter } from './helpers/testRouter.js'
import { calls, resetRequestStub, respond } from './helpers/stubApiRequest.js'

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
}

const BASES = [{ id: 'kb1', name: 'KB1' }]
const FILE_A = { id: 7, name: '员工手册.md', size: 2048, created_at: '2024-01-01T00:00:00Z' }
const FILE_B = { id: 8, name: '差旅制度.md', size: 4096, created_at: '2024-01-02T00:00:00Z' }

const settle = async (view) => {
  await view.flush(8)
  await new Promise((resolve) => setTimeout(resolve, 40))
}

async function until(view, predicate, budgetMs = 2000) {
  const deadline = Date.now() + budgetMs
  for (;;) {
    if (predicate()) return true
    if (Date.now() > deadline) return false
    await view.flush(1)
    await new Promise((resolve) => setTimeout(resolve, 2))
  }
}

// responses 的键是完整 URL；值可以是响应体、Promise（挂起用），
// 处理函数原样交给 stub，Promise 会在 respond 里被 await。
// errors 用于让某个 URL 以 rejected Promise 收场（404 / 网络错误）。
async function mountKnowledge({ route = '/knowledge', responses = {}, errors = {} } = {}) {
  resetRequestStub()
  respond('get', (url) => {
    if (url in errors) throw errors[url]
    if (url in responses) return responses[url]
    if (url === '/knowledge-bases') return BASES
    if (url === '/knowledge') return []
    return {}
  })

  const router = await createTestRouter(route)

  const view = await mountSfc('views/Knowledge.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  return { view, router }
}

// 详情弹窗没有 append-to-body，但视图的 host 也在 body 里；统一在 document 上找，
// 避免将来有人给 el-dialog 加上 append-to-body 后断言面悄悄失效。
// 「内容预览」是详情弹窗独有的区块，知识库新建/重命名弹窗里没有。
function detailDialog() {
  return [...globalThis.document.querySelectorAll('.el-dialog')].find((node) =>
    node.textContent.includes('内容预览')
  )
}

function dialogText() {
  return detailDialog()?.textContent || ''
}

// el-dialog 关闭时 overlay 走 v-show（display:none），节点仍在 DOM 里。
function detailVisible() {
  const dialog = detailDialog()
  if (!dialog) return false
  const overlay = dialog.closest('.el-overlay')
  return !overlay || overlay.style.display !== 'none'
}

function deferred() {
  let resolve
  const promise = new Promise((done) => {
    resolve = done
  })
  return { promise, resolve }
}

// 断言「没有未处理拒绝」需要主动监听：视图把错误收进状态而不是 rethrow 时，
// 测试本身的断言仍可能是绿的，只有 unhandledRejection 才暴露「某处漏了 catch」。
function trackUnhandledRejections() {
  const seen = []
  const handler = (reason) => seen.push(reason)
  process.on('unhandledRejection', handler)
  return {
    seen,
    stop: () => process.off('unhandledRejection', handler),
  }
}

function requestsSince(mark) {
  return calls.slice(mark).map((entry) => entry.args[0])
}

async function close(view) {
  view.elementPlusUnpatch()
  await view.unmount()
}

// ---------------------------------------------------------------------------
// a. 直接访问 / 刷新地址：query 自解释，自动打开目标文件
// ---------------------------------------------------------------------------

test('直接访问 /knowledge?file=7：取元数据与正文并显示该文件，query 保留', async () => {
  const { view, router } = await mountKnowledge({
    route: '/knowledge?file=7',
    responses: {
      '/knowledge/7': FILE_A,
      '/knowledge/7/content': { content: '员工手册全文' },
    },
  })

  try {
    assert.equal(await until(view, () => detailVisible()), true, '深链挂载后应当自动打开详情预览')
    assert.match(dialogText(), /员工手册\.md/, '弹窗标题/详情应当显示目标文件名')
    assert.match(dialogText(), /员工手册全文/, '正文应当来自 getContent')

    const detailCalls = calls.filter((entry) => entry.args[0] === '/knowledge/7')
    assert.equal(detailCalls.length, 1, '不在列表里的目标应当恰好取一次元数据')
    assert.deepEqual(
      detailCalls[0].args[1],
      { silent: true },
      '元数据请求应当静默：不存在/失败由视图统一给文案，不叠加拦截器 toast'
    )
    assert.deepEqual(
      calls.find((entry) => entry.args[0] === '/knowledge/7/content')?.args[1],
      { silent: true },
      '正文请求沿用既有的 silent 约定，失败只落在预览区'
    )
    assert.equal(
      router.currentRoute.value.query.file,
      '7',
      '成功打开后 file query 必须保留：刷新/分享地址仍能恢复到同一个文件'
    )
  } finally {
    await close(view)
  }
})

test('目标已在文件列表里：直接用列表行打开，不再重复请求元数据', async () => {
  const { view, router } = await mountKnowledge({
    route: '/knowledge',
    responses: {
      '/knowledge': [FILE_A],
      '/knowledge/7/content': { content: '列表文件全文' },
    },
  })

  try {
    assert.equal(
      await until(view, () => view.text().includes('员工手册.md')),
      true,
      '前置：文件列表已经渲染'
    )
    const mark = calls.length

    await router.push('/knowledge?file=7')
    assert.equal(await until(view, () => detailVisible()), true, 'query 变化后应当打开详情')
    await settle(view)

    const urls = requestsSince(mark)
    assert.ok(
      !urls.includes('/knowledge/7'),
      '列表里已有该文件时不应再发元数据请求（应直接用列表行）'
    )
    assert.ok(urls.includes('/knowledge/7/content'), '仍然应当读取正文')
    assert.match(dialogText(), /列表文件全文/)
    assert.equal(router.currentRoute.value.query.file, '7')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// b. 文件不存在：友好提示 + 清理 file query（保留其余参数）
// ---------------------------------------------------------------------------

test('文件不存在（404）：友好提示、清掉 file query，不打开空白详情、无未处理拒绝', async () => {
  const notFound = new Error('请求失败')
  notFound.response = { status: 404, data: { detail: '知识库文件不存在' } }
  const unhandled = trackUnhandledRejections()
  const { view, router } = await mountKnowledge({
    route: '/knowledge?file=7&keyword=abc',
    errors: { '/knowledge/7': notFound },
  })

  try {
    assert.equal(
      await until(view, () => view.messages.some((item) => item.level === 'warning')),
      true,
      '文件不存在时应当给用户提示，而不是静默失败'
    )
    assert.deepEqual(
      view.messages,
      [{ level: 'warning', message: '文件不存在或已删除' }],
      '提示文案应当是「文件不存在或已删除」'
    )
    assert.equal(
      await until(view, () => router.currentRoute.value.query.file === undefined),
      true,
      '文件不存在时应当把 file query 清掉，避免地址停在无法恢复的状态'
    )
    assert.equal(
      router.currentRoute.value.query.keyword,
      'abc',
      '清理 file 时不得顺手丢掉其他 query 参数'
    )
    assert.equal(detailVisible(), false, '不存在的文件不应打开一个空白详情弹窗')

    await settle(view)
    assert.deepEqual(unhandled.seen, [], '失败请求应当被视图接住，不得逃逸成未处理拒绝')
  } finally {
    unhandled.stop()
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// c. 失败语义：网络错误 ≠ 文件不存在；正文失败 ≠ 文件不存在
// ---------------------------------------------------------------------------

test('元数据请求网络失败：保留 file query 以便重试，错误文案沿用统一出口', async () => {
  const failure = new Error('network down')
  failure.response = { status: 500, data: { detail: '服务暂时不可用' } }
  const { view, router } = await mountKnowledge({
    route: '/knowledge?file=7',
    errors: { '/knowledge/7': failure },
  })

  try {
    assert.equal(await until(view, () => view.messages.length > 0), true, '网络失败应当给出提示')
    assert.deepEqual(
      view.messages,
      [{ level: 'error', message: '服务暂时不可用' }],
      '非 404 失败不能谎称「文件不存在或已删除」'
    )
    assert.equal(
      router.currentRoute.value.query.file,
      '7',
      '网络失败时保留 file query，用户刷新即可重试'
    )
    assert.equal(detailVisible(), false)
  } finally {
    await close(view)
  }
})

test('元数据成功但正文读取失败：保留文件信息，错误落在预览区，不报「文件不存在」', async () => {
  const contentError = new Error('content failed')
  contentError.response = { status: 500, data: { detail: '正文读取超时' } }
  const { view, router } = await mountKnowledge({
    route: '/knowledge?file=7',
    responses: { '/knowledge/7': FILE_A },
    errors: { '/knowledge/7/content': contentError },
  })

  try {
    assert.equal(
      await until(view, () => dialogText().includes('正文读取超时')),
      true,
      '正文读取失败应当由预览区自己的红字呈现'
    )
    assert.match(dialogText(), /员工手册\.md/, '元数据已经成功，文件信息仍应显示')
    assert.equal(
      view.messages.filter((item) => item.message === '文件不存在或已删除').length,
      0,
      '正文读取失败不得被误报成文件不存在'
    )
    assert.equal(
      router.currentRoute.value.query.file,
      '7',
      '正文失败不应清掉 file query：这个文件明明存在'
    )
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// d. 时序：迟到的旧响应不得打开错误的文件
// ---------------------------------------------------------------------------

test('快速切换目标：迟到的旧元数据响应不得打开错误的文件', async () => {
  const pendingA = deferred()
  const pendingB = deferred()
  const { view, router } = await mountKnowledge({
    route: '/knowledge?file=7',
    responses: {
      '/knowledge/7': pendingA.promise,
      '/knowledge/8': pendingB.promise,
      '/knowledge/8/content': { content: '差旅制度全文' },
    },
  })

  try {
    assert.equal(
      await until(view, () => calls.some((entry) => entry.args[0] === '/knowledge/7')),
      true,
      '前置：7 的元数据请求已经发出并挂起'
    )

    await router.push('/knowledge?file=8')
    assert.equal(
      await until(view, () => calls.some((entry) => entry.args[0] === '/knowledge/8')),
      true,
      '切到 8 后应当发出 8 的元数据请求'
    )

    pendingB.resolve(FILE_B)
    assert.equal(
      await until(view, () => dialogText().includes('差旅制度全文')),
      true,
      '新目标应当正常打开'
    )

    // 旧请求此时才回来。token 已被 query 变化作废，不得把标题/正文改回 7。
    pendingA.resolve(FILE_A)
    await settle(view)

    assert.match(dialogText(), /差旅制度\.md/, '弹窗应当仍指向 8')
    assert.doesNotMatch(dialogText(), /员工手册/, '迟到的 7 不得覆盖已打开的 8')
    assert.equal(
      calls.some((entry) => entry.args[0] === '/knowledge/7/content'),
      false,
      '作废目标的正文请求不应再发出'
    )
    assert.equal(router.currentRoute.value.query.file, '8')
  } finally {
    pendingA.resolve(FILE_A)
    pendingB.resolve(FILE_B)
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// e. 回归：既有「详情」按钮行为不变
// ---------------------------------------------------------------------------

test('列表里的「详情」按钮仍按原路径打开预览，且不产生 file query', async () => {
  const { view, router } = await mountKnowledge({
    route: '/knowledge',
    responses: {
      '/knowledge': [FILE_A],
      '/knowledge/7/content': { content: '列表详情全文' },
    },
  })

  try {
    assert.equal(
      await until(view, () => view.text().includes('员工手册.md')),
      true,
      '前置：文件列表已经渲染'
    )
    const detailButton = view
      .queryAll('button')
      .find((node) => node.textContent.trim() === '详情')
    assert.ok(detailButton, '列表行应当有「详情」按钮')
    detailButton.click()

    assert.equal(
      await until(view, () => dialogText().includes('列表详情全文')),
      true,
      '点击「详情」仍应打开该文件的正文'
    )
    assert.equal(
      router.currentRoute.value.query.file,
      undefined,
      '从列表行打开详情不需要写 file query，地址契约只服务于跨页跳转'
    )
  } finally {
    await close(view)
  }
})
