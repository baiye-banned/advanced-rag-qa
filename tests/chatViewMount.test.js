// Chat.vue 的挂载用例：真 mount 视图，走完一次「提问 -> 流式回答」的主路径（issue #180 验收 2）。
//
// 由来：`git grep Chat.vue tests/` 此前只命中两处，且**都不是挂载** ——
// `chatStore.test.js` 里是一个桩字符串，`test_internal_error_no_echo.py` 是后端用例。
// 视图自己的胶水（`sendMessage` 的清空输入、`handleInputEnter` 的 Enter / Shift 分支
// 与 IME 合成守卫、流式占位的渲染、知识库选中的接线）此前全部只有阅读证据，
// 键盘那几条现在由文件末尾的 d 组用例接管。
//
// 与既有用例的分工：
//   tests/chatStore.test.js  用 registerHooks 换掉 `@/api/chat`，测 **store** 的状态机
//   tests/streamEvents.test.js / chatApi.test.js  测流解析与响应校验的**纯函数**
//   本文件                    视图真的跑起来之后：点「发送」到底发出去了什么、
//                            屏幕上的气泡是谁渲染的、store 被改成了什么
//
// 替身扎在两处**模块/平台边界**上，被测代码照常真跑：
//   api/request.js -> stubApiRequest.js   只换 HTTP 出口（REST 那几个接口）
//   globalThis.fetch                      流式问答走的是 fetch，不是 axios（见 src/api/chat.js），
//                                         所以流那条链路换的是平台出口，chatStore / streamEvents /
//                                         chat.js 全是真货
// 于是「回答上屏」是组件真实响应式状态驱动的 DOM 结果：助手气泡由真的
// MarkdownRenderer.vue 渲染（子组件也是真实 SFC，不是桩件）。
//
// 路由：Chat.vue 用 useRoute/useRouter，挂载时给一个 memory history 的路由。
// 路由指向哪个组件不重要（本文件只挂 Chat.vue 自己），但路由表必须存在，
// 否则 `router.replace('/chat/c1')` 会抛「No match」。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { calls, callsOf, resetRequestStub, respond } from './helpers/stubApiRequest.js'
import { confirmations, resetConfirmStub, respondWith } from './helpers/stubConfirm.js'

// vue-router 必须**动态** import，且排在 vueMount 之后 —— 它自己 import 'vue'，
// 而 vue 的 runtime-dom 在模块求值时就抓走 document
//（`const doc = typeof document !== 'undefined' ? document : null`），
// 静态 import 会让这行跑在 vueMount 装 jsdom 全局之前，症状是挂载时
// 「Cannot read properties of null (reading 'createElement')」。
// 与 vueMount.js 文件头「不要在同一文件里再静态 import vue」是同一条约束。
const { createMemoryHistory, createRouter } = await import('vue-router')

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
  'utils/confirm.js': new URL('./helpers/stubConfirm.js', import.meta.url).href,
}

const KB1 = { id: 'kb1', name: 'KB1' }
const KB2 = { id: 'kb2', name: 'KB2' }
const CONV1 = { id: 'c1', title: '对话一', knowledge_base_id: 'kb1', knowledge_base_name: 'KB1' }

const encoder = new TextEncoder()

// 把若干 SSE 帧做成真的 ReadableStream：readStreamEvents 走的是 getReader()，
// 换成一个数组/字符串都会在解码那一步露馅。
function sseStream(frames) {
  const encoded = frames.map((frame) => encoder.encode(frame))
  return new ReadableStream({
    pull(controller) {
      if (!encoded.length) {
        controller.close()
        return
      }
      controller.enqueue(encoded.shift())
    },
  })
}

// 流式出口的替身：记录每次 fetch，并按用例给的帧序列作答。
function installStreamStub(frames) {
  const streamCalls = []
  globalThis.fetch = async (url, options) => {
    streamCalls.push({ url, options })
    return {
      ok: true,
      body: sseStream(frames),
      async json() {
        return {}
      },
    }
  }
  return streamCalls
}

const ANSWER_FRAMES = [
  'data: {"type":"conversation","conversation":{"id":"c1","title":"新对话","knowledge_base_id":"kb1","knowledge_base_name":"KB1"}}\n\n',
  'data: {"type":"token","content":"你好"}\n\n',
  'data: {"type":"token","content":"，世界"}\n\n',
  'data: [DONE]\n\n',
]

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

function createTestRouter() {
  const blank = { render: () => null }
  return createRouter({
    history: createMemoryHistory(),
    routes: [
      { path: '/', redirect: '/chat' },
      { path: '/chat', name: 'Chat', component: blank },
      { path: '/chat/:id', name: 'ChatDetail', component: blank },
      { path: '/knowledge', component: blank },
      { path: '/login', component: blank },
    ],
  })
}

// 服务端替身：一个会话 + 它已经落库的消息。
// 消息列表**必须**带上刚发出去的那条提问与它的回答：流结束后 store 会按
// `refreshMessages` 重新拉一遍并把结果写回 messages，服务端若返回空列表，
// 屏幕上刚发的气泡会被这次刷新整条抹掉（用例红的会是「用户气泡不在」，而不是被测的接线）。
const PERSISTED = [
  { id: 1, role: 'user', content: '这是一个问题', created_at: '2024-01-01T00:00:01Z' },
  { id: 2, role: 'assistant', content: '你好，世界', created_at: '2024-01-01T00:00:02Z' },
]

async function mountChat({
  bases = [KB1, KB2],
  streamFrames = ANSWER_FRAMES,
  messages = PERSISTED,
  route = '/chat',
  responses = {},
} = {}) {
  resetRequestStub()
  resetConfirmStub()
  respond('get', (url) => {
    if (url === '/knowledge-bases') return bases
    if (url === '/chat/conversations') return [CONV1]
    if (url === '/chat/conversations/c1') return messages
    if (url in responses) return responses[url]
    return {}
  })
  const streamCalls = installStreamStub(streamFrames)

  const router = createTestRouter()
  await router.push(route)
  await router.isReady()

  const view = await mountSfc('views/Chat.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  return { view, router, streamCalls }
}

// 输入并点「发送」。返回点击前的水位线（fetch 的调用数）。
async function sendQuestion(view, text) {
  const textarea = view.query('textarea')
  assert.ok(textarea, '页面上应当有提问用的输入框')
  textarea.value = text
  textarea.dispatchEvent(new Event('input', { bubbles: true }))
  await view.flush(3)

  const button = view.buttonByText('发送')
  assert.ok(button, '页面上应当有「发送」按钮')
  assert.equal(button.disabled, false, '输入内容后「发送」按钮应当可用（否则后面只是空转）')

  const mark = globalThis.__streamCalls.length
  button.click()
  return mark
}

// 在输入框上按一次 Enter。cancelable: true 是必须的：断言面是「守卫有没有走到
// preventDefault」，不可取消的事件恒为 defaultPrevented === false，会让对照组假绿。
async function pressEnter(view, init = {}) {
  const textarea = view.query('textarea')
  assert.ok(textarea, '页面上应当有提问用的输入框')
  const event = new KeyboardEvent('keydown', {
    key: 'Enter', code: 'Enter', bubbles: true, cancelable: true, ...init,
  })
  textarea.dispatchEvent(event)
  await view.flush(3)
  return event
}

// 进入合成态并打入一段「尚未上屏」的文本：先 compositionstart 置 isComposing，
// 再把 DOM 值改掉并以 isComposing 的 input 通知（el-input 此时会早退，v-model 不动）。
async function typeDuringComposition(view, text) {
  const textarea = view.query('textarea')
  textarea.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true }))
  textarea.value = text
  textarea.dispatchEvent(new InputEvent('input', { bubbles: true, isComposing: true, data: text }))
  await view.flush(3)
  return textarea
}

async function close(view) {
  view.elementPlusUnpatch()
  await view.unmount()
}

// ---------------------------------------------------------------------------
// a. 主路径：提问 -> 流式回答上屏
// ---------------------------------------------------------------------------

test('发送消息：问题进入流式请求，回答经 MarkdownRenderer 上屏，输入框清空', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    assert.match(view.text(), /今天想了解什么/, '空会话时应当渲染欢迎区（控制项）')

    const mark = await sendQuestion(view, '这是一个问题')
    await until(view, () => streamCalls.length > mark)
    await until(view, () => view.text().includes('你好'))
    await settle(view)

    // 承重①：流式请求真的发出去了，且带上了问题本身与选中的知识库。
    assert.equal(streamCalls.length - mark, 1, '点「发送」应当恰好发一次流式请求')
    const [{ url, options }] = streamCalls.slice(mark)
    assert.match(url, /\/chat\/stream$/, '流式请求应当打到 /chat/stream')
    const body = JSON.parse(options.body)
    assert.deepEqual(
      { question: body.question, knowledge_base_id: body.knowledge_base_id },
      { question: '这是一个问题', knowledge_base_id: 'kb1' },
      '请求体应当带上问题与当前选中的知识库'
    )

    // 承重②：两条气泡都在屏幕上，助手那条是**真的 MarkdownRenderer** 渲染的
    //（子组件是真实 SFC，`.markdown-body` 是它自己的 v-html 出口）。
    assert.match(view.text(), /这是一个问题/, '用户气泡应当上屏')
    const markdownBodies = view.queryAll('.markdown-body')
    assert.equal(markdownBodies.length, 1, '助手回答应当由 MarkdownRenderer 渲染出一个 .markdown-body')
    assert.match(markdownBodies[0].innerHTML, /你好，世界/, '流式增量应当拼成完整回答')

    // 承重③：输入框被清空（`sendMessage` 里的 `question.value = ''`）。
    // 少了它，用户发完一条还得手删一遍。
    assert.equal(view.query('textarea').value, '', '发送后输入框应当清空')

    // 流结束后按钮回到可用态（streaming 复位），而不是永久转圈。
    await until(view, () => view.buttonByText('发送')?.disabled === true || true)
    assert.equal(view.query('textarea').disabled, false, '流结束后输入框应当恢复可用')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// b. 对照：没有内容时不发请求
// ---------------------------------------------------------------------------

test('空输入：不点得动「发送」，也不会发出流式请求', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    const button = view.buttonByText('发送')
    assert.ok(button, '页面上应当有「发送」按钮')
    // 模板 :273 的禁用条件 `!question.trim() && !attachments.length`：
    // 没有文字也没有图片时按钮是禁用的。
    assert.equal(button.disabled, true, '没有输入内容时「发送」按钮应当禁用')

    button.click()
    await settle(view)
    assert.equal(streamCalls.length, 0, '空输入不得发出流式请求')
    assert.equal(view.queryAll('.markdown-body').length, 0, '没有提问就不该有回答气泡')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// c. 接线：挂载后当前知识库选中与列表同步
// ---------------------------------------------------------------------------

test('挂载后当前知识库与列表第一个同步（syncSelectedKnowledgeBase 接线）', async () => {
  const { view } = await mountChat()

  try {
    // 视图真的拉了知识库列表（挂载期的 GET /knowledge-bases 不是空跑）。
    assert.ok(
      calls.some((entry) => entry.method === 'get' && entry.args[0] === '/knowledge-bases'),
      '挂载时应当拉一次知识库列表'
    )
    // `syncSelectedKnowledgeBase`：没有首选时落到列表第一个，并写回 store。
    // 少了这步，选中停在 null，上面那条「发送时带 knowledge_base_id」的断言也会跟着红。
    assert.match(view.text(), /KB1/, '当前知识库应当显示为列表第一个')
    const selected = view.queryAll('.el-select').length
    assert.ok(selected > 0, '页面上应当渲染出知识库选择器')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// d. 键盘接线：Enter / Shift+Enter / IME 合成态（issue #240）
//
// 这一组把 `handleInputEnter` 的三条出口都钉住：非合成态的 Enter 照常发送、
// Shift+Enter 留给换行、合成态（isComposing 或旧引擎的 keyCode 229）一律放行给输入法。
// 判据是**事件本身**：组字中的 Enter 若被 preventDefault，输入法就没法用它选字
//（这正是 #240 里「候选框被打掉」的机制），所以那两条直接断言 defaultPrevented。
// ---------------------------------------------------------------------------

test('Enter：非合成态按下应当发送并阻止默认行为', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    const textarea = view.query('textarea')
    assert.ok(textarea, '页面上应当有提问用的输入框')
    textarea.value = 'q-plain'
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    await view.flush(3)
    assert.equal(view.buttonByText('发送').disabled, false, '输入内容后「发送」按钮应当可用')

    const event = await pressEnter(view)
    assert.equal(event.defaultPrevented, true, '非合成态的 Enter 应当被 preventDefault（交给 sendMessage）')

    await until(view, () => streamCalls.length > 0)
    assert.equal(streamCalls.length, 1, '非合成态按 Enter 应当恰好发一次流式请求')
    const body = JSON.parse(streamCalls[0].options.body)
    assert.equal(body.question, 'q-plain', '发出的应当就是输入框里的问题')
  } finally {
    await close(view)
  }
})

test('Shift+Enter：非合成态按下不发送、不阻止默认行为', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    const textarea = view.query('textarea')
    textarea.value = 'q-shift'
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    await view.flush(3)

    const event = await pressEnter(view, { shiftKey: true })
    assert.equal(event.defaultPrevented, false, 'Shift+Enter 是换行，不应当被 preventDefault')

    await settle(view)
    assert.equal(streamCalls.length, 0, 'Shift+Enter 不得发出流式请求')
    assert.equal(view.query('textarea').value, 'q-shift', '输入框内容应当留在原地（换行而非清空）')
  } finally {
    await close(view)
  }
})

test('IME：组字中按 Enter 不发送、也不吃掉候选框的默认行为', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    const textarea = await typeDuringComposition(view, 'shijie')
    // 前置闸门（防假绿）：DOM 里已经有未上屏的文本，但 v-model 没跟着走 ——
    // el-input 在 isComposing 的 input 上早退，`question` 仍是空串，按钮因此仍禁用。
    // 少了 vueMount 里的 CompositionEvent 全局，compositionstart 会在 emit 校验器里
    // 抛 ReferenceError 并被 Vue 吞掉，isComposing 无从置起，这条前置会先红。
    assert.equal(textarea.value, 'shijie', '前置：DOM 里应当有未上屏的文本')
    assert.equal(view.buttonByText('发送').disabled, true, '前置：组字中 v-model 不应跟着 DOM 走')

    const event = await pressEnter(view, { isComposing: true })
    assert.equal(event.defaultPrevented, false, '组字中的 Enter 应当留给输入法选字，不得被吃掉')

    await settle(view)
    assert.equal(streamCalls.length, 0, '组字中的 Enter 不得发出流式请求')
  } finally {
    await close(view)
  }
})

test('IME：组字中按 Enter 不得把未上屏文本之外的旧文本发出去（issue #240 原症状）', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    // 先正常输入一段并让它上屏（v-model = '你好'），再进入合成态打 '你好shijie'。
    const textarea = view.query('textarea')
    textarea.value = '你好'
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    await view.flush(3)
    assert.equal(view.buttonByText('发送').disabled, false, '前置：上屏后按钮应当可用')

    await typeDuringComposition(view, '你好shijie')
    assert.equal(textarea.value, '你好shijie', '前置：DOM 里是尚未上屏的新文本')

    const event = await pressEnter(view, { isComposing: true })
    assert.equal(event.defaultPrevented, false, '组字中的 Enter 不得被 preventDefault')

    await settle(view)
    // 修复前的红：这里恰好 1 次请求，且 body.question === '你好' —— 旧文本被发出去、
    // 合成中的候选框被打掉，就是 #240 的字面症状。
    assert.equal(streamCalls.length, 0, '组字中的 Enter 不得发出流式请求（尤其是旧文本）')
  } finally {
    await close(view)
  }
})

test('IME：仅 keyCode 229（旧引擎 / isComposing 缺失）时同样不发送', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    const textarea = view.query('textarea')
    textarea.value = 'q-229'
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    await view.flush(3)
    assert.equal(view.buttonByText('发送').disabled, false, '前置：非空输入时按钮应当可用')

    // 老引擎在「输入法正在处理」时只给 keyCode 229，不给 isComposing。
    const event = await pressEnter(view, { keyCode: 229 })
    assert.equal(event.defaultPrevented, false, '229 也是组字，不得被 preventDefault')

    await settle(view)
    assert.equal(streamCalls.length, 0, '仅凭 keyCode 229 就应当拦下这次 Enter')
  } finally {
    await close(view)
  }
})

test('IME：compositionend 之后按 Enter 发送的是上屏后的全文', async () => {
  const { view, streamCalls } = await mountChat()
  globalThis.__streamCalls = streamCalls

  try {
    const textarea = await typeDuringComposition(view, '你好shijie')
    textarea.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true }))
    await view.flush(3)
    // 上屏后 el-input 会把合成结果补进 v-model（handleCompositionEnd -> handleInput）。
    assert.equal(view.buttonByText('发送').disabled, false, '上屏后按钮应当可用')

    const event = await pressEnter(view)
    assert.equal(event.defaultPrevented, true, '上屏之后的 Enter 应当照常被 sendMessage 接管')

    await until(view, () => streamCalls.length > 0)
    assert.equal(streamCalls.length, 1, '上屏后的 Enter 应当恰好发一次流式请求')
    const body = JSON.parse(streamCalls[0].options.body)
    assert.equal(body.question, '你好shijie', '发出的应当是上屏后的全文')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// e. 消息反馈（issue #250）：👍/👎 的选中态、请求体、取消与失败回滚
//
// 视图这一层只有一条逻辑：`onFeedback` 把「再点同向 = 取消（归 0）」算出来，
// 交给 store 的 setMessageFeedback，并接住 store 回滚后重新抛出的 rejection。
// 所以这里钉三件事：按钮真的渲染出来了、点下去发出去的请求体对不对、
// 失败后 DOM 有没有跟着 store 回到调用前的值。
//
// 失败提示（ElMessage）**不在**断言面内：整个 `api/request.js` 被换成了替身，
// 弹提示的 axios 拦截器（src/api/request.js）根本没进模块图，替身也不负责弹。
// 视图只保证「接住了 rejection、不让它变成未处理的 Promise」。
// ---------------------------------------------------------------------------

// 挂载用例的 views/Chat.vue 与测试这里 import 的是同一个 src/stores/chat.js
//（同一个解析结果 -> 同一个模块实例），所以能从同一个 pinia 里取到视图正在用的 store。
async function viewStore(view) {
  const { useChatStore } = await import(new URL('../src/stores/chat.js', import.meta.url).href)
  return useChatStore(view.pinia)
}

test('反馈按钮：助手消息上有 👍/👎，落库消息可点', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    // PERSISTED 里只有一条助手消息（id=2），所以恰好一组反馈按钮。
    const ups = view.queryAll('[data-testid="feedback-up"]')
    const downs = view.queryAll('[data-testid="feedback-down"]')
    assert.equal(ups.length, 1, '助手消息上应当有一个「有帮助」按钮')
    assert.equal(downs.length, 1, '助手消息上应当有一个「没帮助」按钮')
    // 初始未反馈：两个都用「未选中」色，且因为消息已落库（有 id）都可点。
    for (const button of [...ups, ...downs]) {
      assert.equal(button.classList.contains('text-slate-400'), true, '未反馈时应当用未选中色')
      assert.equal(button.disabled, false, '已落库的助手消息，反馈按钮应当可点')
    }
  } finally {
    await close(view)
  }
})

test('反馈按钮：点「有帮助」发出 feedback=1，按钮切到选中色', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    assert.equal(callsOf('post').length, 0, '前置：挂载后还没有任何 POST')
    const up = view.query('[data-testid="feedback-up"]')
    assert.ok(up, '页面上应当有「有帮助」按钮')

    up.click()
    await view.flush(2)

    // 承重①：请求打到消息级反馈路由，body 是 { feedback: 1 }。
    assert.deepEqual(callsOf('post'), [
      { method: 'post', args: ['/chat/messages/2/feedback', { feedback: 1 }] },
    ], '点「有帮助」应当向 /chat/messages/2/feedback 发 { feedback: 1 }')

    // 承重②：选中态上屏（乐观更新，不等请求回来）。
    assert.equal(up.classList.contains('text-brand-600'), true, '选中后应当用高亮色')
    assert.equal(up.classList.contains('text-slate-400'), false, '选中后不应再是未选中色')
  } finally {
    await close(view)
  }
})

test('反馈按钮：再点同向取消（feedback=0），选中态回落', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const up = view.query('[data-testid="feedback-up"]')
    up.click()
    await view.flush(2)
    assert.equal(up.classList.contains('text-brand-600'), true, '前置：第一次点应当选中')

    up.click()
    await view.flush(2)

    // 「再点同向 = 取消」：第二次发出的是 0，而不是把 1 再发一遍。
    assert.deepEqual(
      callsOf('post').map((entry) => entry.args[1]),
      [{ feedback: 1 }, { feedback: 0 }],
      '第二次点击应当发 { feedback: 0 }（取消），而不是重复提交 1'
    )
    assert.equal(up.classList.contains('text-slate-400'), true, '取消后应当回到未选中色')
  } finally {
    await close(view)
  }
})

test('反馈按钮：提交失败时回滚到调用前的值，并接住 rejection', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)
    respond('post', () => {
      throw new Error('反馈提交失败')
    })
    const up = view.query('[data-testid="feedback-up"]')

    up.click()
    await view.flush(2)
    // 先乐观选中，请求被拒后滚回去。
    await settle(view)

    assert.equal(up.classList.contains('text-slate-400'), true, '失败后按钮应当回到未选中色')
    const message = store.messages.find((item) => item.id === 2)
    assert.equal(message.feedback, 0, '失败后 store 里的值应当回滚到调用前的 0')
    // 请求确实发出去了（不是因为没发才没变）。
    assert.equal(callsOf('post').length, 1, '无论成败，请求都应当发出去一次')
  } finally {
    await close(view)
  }
})

test('反馈按钮：没有 id 的本地消息按钮禁用，且点击不发请求', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    // 本地/流式中的助手消息没有 id（normalizeMessage 把缺省 id 归一成 null），
    // 此时按钮必须禁用 —— 否则会发出 `/chat/messages/null/feedback` 这种打不中的请求。
    const store = await viewStore(view)
    store.addMessage({ role: 'assistant', content: '本地回答', isLocal: true })
    await view.flush(2)

    const localUp = view.queryAll('[data-testid="feedback-up"]').at(-1)
    assert.ok(localUp, '本地助手消息也应当渲染出反馈按钮')
    assert.equal(localUp.disabled, true, '没有 id 的消息，反馈按钮应当禁用')

    localUp.click()
    await view.flush(2)
    assert.equal(callsOf('post').length, 0, '禁用的反馈按钮不得发出任何请求')
  } finally {
    await close(view)
  }
})

// f. 会话导出（issue #252）：走真实按钮、chatAPI 和下载 helper，只替换 HTTP 与浏览器下载出口。
function recordExportDownloads(t) {
  const downloads = []
  const urls = []
  t.mock.method(URL, 'createObjectURL', (blob) => {
    urls.push({ blob, revoked: false })
    return `blob:export-${urls.length}`
  })
  t.mock.method(URL, 'revokeObjectURL', (url) => {
    const index = Number(url.split('-').at(-1)) - 1
    urls[index].revoked = true
  })
  t.mock.method(HTMLAnchorElement.prototype, 'click', function () {
    downloads.push({ href: this.href, filename: this.download })
  })
  return { downloads, urls }
}

function exportRequests() {
  return callsOf('get').filter((entry) => entry.args[0].endsWith('/export'))
}

test('导出按钮：未保存的新会话不可点，已保存的空会话可点，生成期间禁用', async (t) => {
  const { downloads } = recordExportDownloads(t)
  const { view } = await mountChat({ messages: [] })
  try {
    const button = view.query('[data-testid="export-markdown"]')
    assert.ok(button, '顶部应当有导出按钮')
    assert.equal(button.disabled, true, '新会话尚无 id 时不可导出')
    button.click()
    await view.flush(2)
    assert.equal(exportRequests().length, 0)

    const store = await viewStore(view)
    await store.selectConversation('c1')
    await view.flush(3)
    assert.equal(store.messages.length, 0)
    assert.equal(button.disabled, false, '已保存的空会话也允许导出标题')

    store.streamingConversationId = 'c1'
    store.streaming = true
    await view.flush(2)
    assert.equal(button.disabled, true, '回答生成期间不可导出')
    button.click()
    await view.flush(2)
    assert.equal(exportRequests().length, 0)
    assert.equal(downloads.length, 0)
    store.streaming = false
  } finally {
    await close(view)
  }
})

for (const { headers, filename } of [
  {
    headers: {
      'content-disposition': `attachment; filename="conversation.md"; filename*=UTF-8''${encodeURIComponent('服务端标题.md')}`,
    },
    filename: '服务端标题.md',
  },
  { headers: {}, filename: '对话一.md' },
]) {
  test(`导出按钮：防重复点击，切换会话后仍下载点击时的会话（${filename}）`, async (t) => {
    const { downloads, urls } = recordExportDownloads(t)
    const { view, router } = await mountChat({ route: '/chat/c1' })
    let resolveExport
    const pendingExport = new Promise((resolve) => { resolveExport = resolve })
    respond('get', (url) => {
      if (url === '/chat/conversations/c1/export') return pendingExport
      if (url === '/knowledge-bases') return [KB1, KB2]
      if (url === '/chat/conversations/c2') return []
      return []
    })
    try {
      const store = await viewStore(view)
      const initialMessages = JSON.stringify(store.messages)
      store.conversations.push({ ...CONV1, id: 'c2', title: '对话二' })
      const button = view.query('[data-testid="export-markdown"]')
      button.click()
      button.click()
      await view.flush(2)
      assert.equal(exportRequests().length, 1, '快速点击两次只发一个请求')
      assert.deepEqual(exportRequests()[0].args, [
        '/chat/conversations/c1/export',
        { responseType: 'blob', returnFullResponse: true, silent: true },
      ])
      assert.equal(button.disabled, true, '导出请求在途期间按钮禁用')
      assert.equal(JSON.stringify(store.messages), initialMessages, '导出不修改原消息')

      await router.push('/chat/c2')
      await view.flush(4)
      assert.equal(store.currentId, 'c2', '前置：已经切换到了另一会话')
      assert.equal(button.disabled, true)
      const blob = new Blob(['# 对话一\n\n完整记录'], { type: 'text/markdown;charset=utf-8' })
      resolveExport({ status: 200, data: blob, headers })
      await settle(view)
      assert.deepEqual(downloads, [{ href: 'blob:export-1', filename }])
      assert.strictEqual(urls[0].blob, blob, '实际保存后端原始文件')
      assert.equal(urls[0].revoked, true, '下载完成后释放临时 URL')
      assert.equal(document.querySelector('a[download]'), null, '临时链接应已移除')
      assert.equal(button.disabled, false, '结束后可导出当前会话')
      assert.deepEqual(view.messages, [])
    } finally {
      await close(view)
    }
  })
}

// ---------------------------------------------------------------------------
// g. 会话切换失败（issue #261）：路由回滚，且不留下「标识换了、内容没换」的错位
// ---------------------------------------------------------------------------

test('切换会话加载失败：提示错误并把路由回滚到原会话，消息保持原样', async () => {
  const { view, router } = await mountChat({ route: '/chat/c1' })
  // 挂载后再换 handler：c2 的消息请求失败，其余接口照常。
  respond('get', (url) => {
    if (url === '/knowledge-bases') return [KB1, KB2]
    if (url === '/chat/conversations') return [CONV1, { ...CONV1, id: 'c2', title: '对话二' }]
    if (url === '/chat/conversations/c1') return PERSISTED
    if (url === '/chat/conversations/c2') throw new Error('network down')
    return {}
  })

  try {
    const store = await viewStore(view)
    const initialMessages = JSON.stringify(store.messages)
    assert.equal(store.currentId, 'c1', '前置：原会话已经加载')
    assert.deepEqual(store.messages.map((item) => item.content), ['这是一个问题', '你好，世界'])

    await router.push('/chat/c2')
    await settle(view)

    // 视图只提示一次，并把地址退回发起切换前的会话。
    assert.deepEqual(view.messages, [{ level: 'error', message: '对话加载失败' }])
    assert.equal(router.currentRoute.value.path, '/chat/c1', '失败后路由应当回滚到原会话')
    // store 里的标识与内容都没有动：不会出现「标题是对话二、消息还是对话一」。
    assert.equal(store.currentId, 'c1', '失败后仍指向原会话')
    assert.equal(JSON.stringify(store.messages), initialMessages, '原会话消息不得被清空或替换')
    assert.equal(store.loading, false, '失败后不应一直停在加载态')
    assert.match(view.text(), /你好，世界/, '屏幕上仍是原会话的内容')
  } finally {
    await close(view)
  }
})

test('导出按钮：HTTP 失败只提示一次，不保存文件且恢复可重试', async (t) => {
  const { downloads, urls } = recordExportDownloads(t)
  const { view } = await mountChat({ route: '/chat/c1' })
  respond('get', () => {
    const error = new Error('Request failed')
    error.response = { status: 404, data: new Blob(['{"detail":"对话不存在"}'], { type: 'application/json' }) }
    throw error
  })
  try {
    const store = await viewStore(view)
    const initialMessages = JSON.stringify(store.messages)
    const button = view.query('[data-testid="export-markdown"]')
    button.click()
    await settle(view)
    assert.equal(exportRequests().length, 1)
    assert.deepEqual(view.messages, [{ level: 'error', message: '对话不存在或无权导出' }])
    assert.equal(downloads.length, 0)
    assert.equal(urls.length, 0)
    assert.equal(button.disabled, false)
    assert.equal(JSON.stringify(store.messages), initialMessages)
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// h. 深链加载在飞时流认领新会话：迟到的加载作废后，路由要跟随实际会话
//
// issue #261 返修：上一轮加的 claimedAtStart 守卫挡住了迟到加载对
// currentId / messages 的覆盖，但 Chat.vue 的路由 watcher 在
// `await selectConversation(c1)` 返回后没有校正路由。深链 /chat/c1 时
// 两个负责跳转的 watcher 都以 `!route.params.id` 为前提，URL 会一直停在
// c1 —— 界面与刷新都会把用户带回旧会话。本用例从挂载开始走完整链路：
// 加载在飞 -> 提问 -> 流认领 c9 -> c1 加载返回 -> 路由校正到 c9。
// ---------------------------------------------------------------------------

const C1_OLD_HISTORY = [
  { id: 11, role: 'user', content: 'c1 的旧问题', created_at: '2024-01-01T00:00:01Z' },
  { id: 12, role: 'assistant', content: 'c1 的旧回答', created_at: '2024-01-01T00:00:02Z' },
]

const C9_PERSISTED = [
  { id: 91, role: 'user', content: '在途时提问', created_at: '2024-01-01T00:00:03Z' },
  { id: 92, role: 'assistant', content: '新回答', created_at: '2024-01-01T00:00:04Z' },
]

const CLAIM_FRAMES = [
  'data: {"type":"conversation","conversation":{"id":"c9","title":"新对话","knowledge_base_id":"kb1","knowledge_base_name":"KB1"}}\n\n',
  'data: {"type":"token","content":"新回答"}\n\n',
  'data: [DONE]\n\n',
]

test('深链加载在飞时流认领新会话：迟到的加载作废后路由校正到新会话', async () => {
  let resolveC1
  // 用挂起的 Promise 当 c1 的消息响应：请求真的发出去，但由用例决定何时返回。
  const pendingC1 = new Promise((resolve) => { resolveC1 = resolve })
  const { view, router, streamCalls } = await mountChat({
    route: '/chat/c1',
    messages: pendingC1,
    streamFrames: CLAIM_FRAMES,
    responses: { '/chat/conversations/c9': C9_PERSISTED },
  })
  globalThis.__streamCalls = streamCalls

  try {
    const store = await viewStore(view)
    // 前置：深链 c1 的加载已经发出且在飞，currentId 仍为空。
    await until(view, () => calls.some(
      (entry) => entry.method === 'get' && entry.args[0] === '/chat/conversations/c1'
    ))
    assert.equal(store.currentId, null, '前置：c1 的加载还没返回，currentId 仍为空')
    assert.equal(store.loadingConversationId, 'c1', '前置：c1 的加载在飞')

    // 加载在飞时提问：流为新建的会话 c9 投递 conversation 事件并认领它。
    await sendQuestion(view, '在途时提问')
    await until(view, () => store.currentId === 'c9')
    assert.equal(store.currentId, 'c9', '流应当先认领新会话 c9')
    assert.equal(
      router.currentRoute.value.path,
      '/chat/c1',
      '前置：旧加载尚未返回，路由仍停在深链会话（否则本用例测不到校正）'
    )

    // c1 的旧加载这时才返回：store 必须作废它，视图必须把路由校正到 c9。
    resolveC1(C1_OLD_HISTORY)
    await settle(view)
    await until(view, () => router.currentRoute.value.path === '/chat/c9')

    assert.equal(store.currentId, 'c9', 'currentId 应当仍是流认领的 c9')
    assert.equal(router.currentRoute.value.path, '/chat/c9', '路由应当校正到实际生效的会话 c9')
    assert.deepEqual(
      store.messages.map((item) => item.content),
      ['在途时提问', '新回答'],
      '刚发出的提问及其回答应当留在消息列表'
    )
    assert.ok(
      !store.messages.some((item) => String(item.content).includes('c1 的旧')),
      'c1 的旧历史不得覆盖当前视图'
    )
    assert.doesNotMatch(view.text(), /c1 的旧回答/, '屏幕上不得出现 c1 的旧历史')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// i. 切换会话加载态（issue #254）
//
// #261 已在 store 里给出 loadingConversationId，但视图此前没用：切会话的等待期
// 要么露出上一个会话的旧消息，要么从空白页闪一下「今天想了解什么？」。
// 这一组直接把 GET 挂起，在等待窗口内断言主区的加载遮罩（.el-loading-mask）
// 与 aria-busy，而不是读源码字符串。
// ---------------------------------------------------------------------------

const C2_HISTORY = [
  { id: 21, role: 'user', content: 'c2 的问题', created_at: '2024-01-01T00:00:05Z' },
  { id: 22, role: 'assistant', content: 'c2 的回答', created_at: '2024-01-01T00:00:06Z' },
]

function deferredResponse() {
  let resolve
  const promise = new Promise((done) => {
    resolve = done
  })
  return { promise, resolve }
}

// Element Plus 关闭遮罩时先 v-show 隐藏，400ms 后才把节点摘掉；
// 所以「加载态消失」要按可见性判断，不能只查节点是否存在。
function loadingMaskShown(view) {
  const mask = view.query('.el-loading-mask')
  return Boolean(mask) && mask.style.display !== 'none'
}

test('从有消息的会话切到另一会话：等待期显示加载态，不显示空态与旧会话内容', async () => {
  const pending = deferredResponse()
  const { view, router } = await mountChat({
    route: '/chat/c1',
    responses: { '/chat/conversations/c2': pending.promise },
  })
  let nav

  try {
    const store = await viewStore(view)
    assert.equal(store.currentId, 'c1', '前置：A 已经加载完成')
    assert.match(view.text(), /你好，世界/, '前置：A 的消息已经渲染')
    assert.equal(loadingMaskShown(view), false, '前置：没有加载遮罩')

    nav = router.push('/chat/c2')
    assert.equal(
      await until(view, () => store.loadingConversationId === 'c2'),
      true,
      '切到 B 之后 store 应当处于加载 B 的状态'
    )
    await view.flush(2)

    assert.equal(loadingMaskShown(view), true, '等待期主区应当显示加载遮罩')
    assert.equal(view.query('main').getAttribute('aria-busy'), 'true', '等待期主区应当标记 aria-busy')
    assert.doesNotMatch(view.text(), /今天想了解什么/, '等待期不得显示空态')
    assert.doesNotMatch(view.text(), /你好，世界/, 'A 的旧消息不得被当成 B 的内容展示')

    pending.resolve(C2_HISTORY)
    assert.equal(
      await until(view, () => store.loadingConversationId === null),
      true,
      'B 返回后加载态应当结束'
    )
    assert.equal(await until(view, () => view.text().includes('c2 的回答')), true)

    assert.equal(store.currentId, 'c2', '加载成功后提交 B 的会话 id')
    assert.equal(view.query('main').getAttribute('aria-busy'), 'false')
    assert.equal(await until(view, () => !loadingMaskShown(view)), true, '加载结束后遮罩应当隐藏')
    assert.match(view.text(), /c2 的问题/, '应当渲染 B 的用户消息')
    assert.match(view.text(), /c2 的回答/, '应当渲染 B 的助手消息')
    assert.doesNotMatch(view.text(), /你好，世界/, 'A 的内容不得留在 B 的视图里')
  } finally {
    pending.resolve([])
    await nav
    await close(view)
  }
})

test('从空白页切到会话：等待期显示加载态，不闪空态', async () => {
  const pending = deferredResponse()
  const { view, router } = await mountChat({
    route: '/chat',
    responses: { '/chat/conversations/c2': pending.promise },
  })
  let nav

  try {
    const store = await viewStore(view)
    assert.equal(store.currentId, null, '前置：空白页没有当前会话')
    assert.match(view.text(), /今天想了解什么/, '前置：空白页显示空态')

    nav = router.push('/chat/c2')
    assert.equal(
      await until(view, () => store.loadingConversationId === 'c2'),
      true,
      '切到会话后应当进入加载态'
    )
    await view.flush(2)

    assert.equal(loadingMaskShown(view), true, '等待期主区应当显示加载遮罩')
    assert.equal(view.query('main').getAttribute('aria-busy'), 'true')
    assert.doesNotMatch(view.text(), /今天想了解什么/, '等待期不得闪出空态')

    pending.resolve(C2_HISTORY)
    assert.equal(await until(view, () => view.text().includes('c2 的回答')), true)
    assert.equal(await until(view, () => !loadingMaskShown(view)), true, '加载结束后遮罩应当隐藏')
    assert.match(view.text(), /c2 的回答/)
  } finally {
    pending.resolve([])
    await nav
    await close(view)
  }
})

test('切到的会话返回空消息：加载完成后仍显示空态', async () => {
  const pending = deferredResponse()
  const { view, router } = await mountChat({
    route: '/chat',
    responses: { '/chat/conversations/c2': pending.promise },
  })
  let nav

  try {
    const store = await viewStore(view)
    nav = router.push('/chat/c2')
    assert.equal(await until(view, () => store.loadingConversationId === 'c2'), true)
    assert.doesNotMatch(view.text(), /今天想了解什么/, '加载中不显示空态')

    pending.resolve([])
    assert.equal(await until(view, () => store.loadingConversationId === null), true)
    await settle(view)

    assert.equal(store.currentId, 'c2', '空会话也应当提交 id')
    assert.equal(store.messages.length, 0)
    assert.match(view.text(), /今天想了解什么/, 'B 确实为空时应当显示空态')
    assert.equal(await until(view, () => !loadingMaskShown(view)), true, '加载结束后遮罩应当隐藏')
  } finally {
    pending.resolve([])
    await nav
    await close(view)
  }
})

test('切到的会话返回消息：加载完成后正常渲染该会话', async () => {
  const pending = deferredResponse()
  const { view, router } = await mountChat({
    route: '/chat',
    responses: { '/chat/conversations/c2': pending.promise },
  })
  let nav

  try {
    const store = await viewStore(view)
    nav = router.push('/chat/c2')
    assert.equal(await until(view, () => store.loadingConversationId === 'c2'), true)

    pending.resolve(C2_HISTORY)
    assert.equal(
      await until(view, () => store.loadingConversationId === null && view.text().includes('c2 的回答')),
      true
    )

    assert.equal(store.currentId, 'c2')
    assert.deepEqual(
      store.messages.map((item) => item.content),
      ['c2 的问题', 'c2 的回答'],
      'store 应当持有 B 的消息'
    )
    assert.match(view.text(), /c2 的问题/)
    assert.match(view.text(), /c2 的回答/)
    assert.equal(view.queryAll('.markdown-body').length, 1, 'B 的助手消息应当经 MarkdownRenderer 渲染')
    assert.doesNotMatch(view.text(), /今天想了解什么/)
    assert.equal(await until(view, () => !loadingMaskShown(view)), true, '加载结束后遮罩应当隐藏')
  } finally {
    pending.resolve([])
    await nav
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// j. 参考资料卡片跳知识库详情（issue #255）
//
// 参考抽屉里的卡片不是永远可点：`file_id` 是跨页跳转的唯一凭据，只有正整数
// 才存在一个可寻址的 `/knowledge?file=<id>` 目标。缺失 / 0 / 非数字时必须保持
// 原来的静态展示，否则用户会点到一个注定「不存在」的详情页。
//
// 抽屉由 element-plus append 到 body（append-to-body），因此断言面在 document 上，
// 不在 view.host 里。关闭抽屉时 overlay 走 v-show，过渡结束后 display:none。
// ---------------------------------------------------------------------------

const SOURCED_MESSAGES = [
  { id: 1, role: 'user', content: '问题', created_at: '2024-01-01T00:00:01Z' },
  {
    id: 2,
    role: 'assistant',
    content: '回答',
    created_at: '2024-01-01T00:00:02Z',
    sources: [{ index: 1, file_id: 7, file_name: '员工手册.md', content: '片段内容' }],
  },
]

const INVALID_ID_MESSAGES = [
  { id: 1, role: 'user', content: '问题', created_at: '2024-01-01T00:00:01Z' },
  {
    id: 2,
    role: 'assistant',
    content: '回答',
    created_at: '2024-01-01T00:00:02Z',
    sources: [
      { index: 1, file_id: 0, file_name: '零号资料.md', content: '片段一' },
      { index: 2, file_name: '没有 id 的资料.md', content: '片段二' },
      { index: 3, file_id: 'abc', file_name: '非数字 id.md', content: '片段三' },
    ],
  },
]

function sourceDrawer() {
  return [...globalThis.document.querySelectorAll('.el-drawer')].find((node) =>
    node.textContent.includes('参考资料')
  )
}

function sourceCards() {
  const drawer = sourceDrawer()
  return drawer ? [...drawer.querySelectorAll('article')] : []
}

function drawerHidden(drawer) {
  const overlay = drawer?.closest('.el-overlay')
  return !overlay || overlay.style.display === 'none'
}

test('参考资料卡片：file_id 有效时点击跳转 /knowledge?file=<id> 并收起抽屉', async () => {
  const { view, router } = await mountChat({ route: '/chat/c1', messages: SOURCED_MESSAGES })

  try {
    const trigger = view.buttonByText('参考 1 篇资料')
    assert.ok(trigger, '助手消息应当渲染参考资料入口')
    trigger.click()
    assert.equal(await until(view, () => sourceCards().length === 1), true, '点击入口后应当打开参考资料抽屉')

    const [card] = sourceCards()
    assert.equal(card.getAttribute('role'), 'button', '有效 file_id 的卡片应当有按钮语义')
    assert.equal(card.getAttribute('tabindex'), '0', '有效 file_id 的卡片应当能通过键盘聚焦')
    assert.equal(card.getAttribute('aria-label'), '查看资料详情：员工手册.md')

    card.click()
    assert.equal(
      await until(view, () => router.currentRoute.value.fullPath === '/knowledge?file=7'),
      true,
      '点击卡片应当按地址契约跳到 /knowledge?file=7'
    )
    assert.equal(router.currentRoute.value.query.file, '7', '目标文件 id 应当体现在可刷新/可分享的 query 上')
    assert.equal(await until(view, () => drawerHidden(sourceDrawer())), true, '跳转后参考资料抽屉应当收起')
  } finally {
    await close(view)
  }
})

test('参考资料卡片：键盘 Enter 与点击等价，空格同样可触发', async () => {
  const { view, router } = await mountChat({ route: '/chat/c1', messages: SOURCED_MESSAGES })

  try {
    view.buttonByText('参考 1 篇资料').click()
    assert.equal(await until(view, () => sourceCards().length === 1), true)

    // 先测空格：卡片是 role=button 的自定义元素，浏览器不会替它把 Space 转成 click，
    // 必须由视图自己接住；只测 Enter 会漏掉这个分支。
    const [card] = sourceCards()
    const space = new KeyboardEvent('keydown', {
      key: ' ', code: 'Space', bubbles: true, cancelable: true,
    })
    card.dispatchEvent(space)
    assert.equal(space.defaultPrevented, true, '空格被卡片处理时应当 preventDefault（避免页面滚动）')
    assert.equal(
      await until(view, () => router.currentRoute.value.fullPath === '/knowledge?file=7'),
      true,
      '空格应当与点击一样跳到详情地址'
    )
  } finally {
    await close(view)
  }
})

test('参考资料卡片：file_id 为 0/缺失/非数字时不可点击，也不跳转', async () => {
  const { view, router } = await mountChat({ route: '/chat/c1', messages: INVALID_ID_MESSAGES })

  try {
    const trigger = view.buttonByText('参考 3 篇资料')
    assert.ok(trigger, '三条来源应当渲染为一个参考资料入口')
    trigger.click()
    assert.equal(await until(view, () => sourceCards().length === 3), true, '抽屉里应当渲染三张卡片')

    for (const card of sourceCards()) {
      assert.equal(card.getAttribute('role'), null, '无效 file_id 的卡片不应有按钮语义')
      assert.equal(card.getAttribute('tabindex'), null, '无效 file_id 的卡片不应进入键盘 Tab 序列')
      assert.equal(card.getAttribute('aria-label'), null, '无效 file_id 的卡片不应声称可以查看详情')
      assert.doesNotMatch(card.className, /cursor-pointer/, '无效 file_id 的卡片不应看起来可点击')

      card.click()
      card.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true })
      )
    }
    await settle(view)

    assert.equal(router.currentRoute.value.fullPath, '/chat/c1', '无效 file_id 不应发生任何跳转')
    assert.equal(drawerHidden(sourceDrawer()), false, '点击无效卡片不应收起抽屉（它本来就没有行为）')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// k. 删除单条消息（issue #256）：确认文案、按角色删除、流式与本地消息守卫
//
// 视图这一层负责三件事：入口的可见/可用性、先确认再发请求、把 store 的结果
// 留在屏幕上。删除语义（用户消息截断、助手消息单删）由 store 用例与后端用例钉住，
// 这里只验证点击真的走到 `DELETE /chat/messages/{id}`，且取消时一个请求都不发。
// ---------------------------------------------------------------------------

function deleteButtons(view, role) {
  return view.queryAll(`[data-testid="delete-message"][data-message-role="${role}"]`)
}

test('删除按钮：用户与助手消息各有一个入口，落库后可点', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const buttons = view.queryAll('[data-testid="delete-message"]')
    assert.equal(buttons.length, 2, 'PERSISTED 的两条消息各应有一个删除入口')
    assert.equal(deleteButtons(view, 'user').length, 1, '用户消息应当有删除入口')
    assert.equal(deleteButtons(view, 'assistant').length, 1, '助手消息应当有删除入口')
    for (const button of buttons) {
      assert.equal(button.disabled, false, '已落库的消息，删除入口应当可点')
    }
  } finally {
    await close(view)
  }
})

test('删除消息：取消确认不调用删除接口，消息原样保留', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)
    respondWith('cancel')
    const button = deleteButtons(view, 'assistant')[0]
    button.click()
    await settle(view)

    // 承重：确认框文案按角色区分，且取消后没有任何 DELETE。
    assert.deepEqual(confirmations, [{ message: '删除这条回答？', title: '删除消息' }])
    assert.equal(callsOf('delete').length, 0, '取消确认不得发出删除请求')
    assert.equal(store.messages.length, 2, '取消后本地消息应当原样保留')
    assert.match(view.text(), /你好，世界/, '取消后助手气泡应当还在屏幕上')
  } finally {
    await close(view)
  }
})

test('删除消息：确认后助手消息只删一条，并刷新会话列表', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)
    const conversationRefreshes = () =>
      callsOf('get').filter((entry) => entry.args[0] === '/chat/conversations').length
    const refreshesBefore = conversationRefreshes()

    deleteButtons(view, 'assistant')[0].click()
    await settle(view)

    assert.deepEqual(confirmations, [{ message: '删除这条回答？', title: '删除消息' }])
    assert.deepEqual(callsOf('delete'), [
      { method: 'delete', args: ['/chat/messages/2'] },
    ], '助手消息应当只删目标这一条')
    assert.deepEqual(store.messages.map((message) => message.id), [1], '本地只应剩下提问')
    assert.doesNotMatch(view.text(), /你好，世界/, '被删的助手气泡应当从屏幕上消失')
    assert.equal(deleteButtons(view, 'assistant').length, 0)
    assert.equal(
      conversationRefreshes(),
      refreshesBefore + 1,
      '删除成功后应当刷新会话列表，让 updated_at 排序生效'
    )
  } finally {
    await close(view)
  }
})

test('删除消息：确认后用户消息连同其后的消息一起截断，并保留当前会话', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)

    deleteButtons(view, 'user')[0].click()
    await settle(view)

    // 文案必须说清「及其后的全部消息」，不能只说删这一条。
    assert.deepEqual(confirmations, [{ message: '删除该提问及其后的全部消息？', title: '删除消息' }])
    assert.deepEqual(callsOf('delete'), [
      { method: 'delete', args: ['/chat/messages/1'] },
    ])
    assert.equal(store.messages.length, 0, '提问及其后的回答都应当从本地消失')
    assert.equal(store.currentId, 'c1', '删空后应保留当前会话（空态），而不是切回新对话')
    assert.equal(view.queryAll('[data-testid="delete-message"]').length, 0)
  } finally {
    await close(view)
  }
})

test('删除消息：删除失败时消息留在屏幕上，且不弹成功提示', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)
    respond('delete', () => {
      throw new Error('删除失败')
    })

    deleteButtons(view, 'assistant')[0].click()
    await settle(view)

    // 请求发出一次、失败被视图接住（未处理 rejection 会让 node --test 直接失败）。
    assert.equal(callsOf('delete').length, 1, '无论成败，删除请求都应当发出一次')
    assert.deepEqual(store.messages.map((message) => message.id), [1, 2], '失败后本地消息不变')
    assert.match(view.text(), /你好，世界/, '失败后气泡应当留在屏幕上')
  } finally {
    await close(view)
  }
})

test('删除按钮：没有 id 的本地消息禁用，且点击不发请求', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)
    store.addMessage({ role: 'assistant', content: '本地回答', isLocal: true })
    await view.flush(2)

    const localButton = deleteButtons(view, 'assistant').at(-1)
    assert.ok(localButton, '本地助手消息也应当渲染出删除入口')
    assert.equal(localButton.disabled, true, '没有 id 的消息不可删除')

    // 绕过 disabled 的点击抑制，直接派发：视图自己的守卫也必须挡住。
    localButton.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await view.flush(2)
    assert.equal(confirmations.length, 0, '不可删除的消息不得弹出确认框')
    assert.equal(callsOf('delete').length, 0, '不可删除的消息不得发出请求')
  } finally {
    await close(view)
  }
})

test('删除消息：确认框打开后开始流式，不报成功并保留消息', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    const store = await viewStore(view)
    respondWith('confirm')

    // 点击同步打开确认框（记录会立刻落下），await 的续体要等当前栈结束才跑。
    deleteButtons(view, 'assistant')[0].click()
    // 确认框还开着时用户开始了生成：store 的删除入口会直接返回 false、不发请求。
    store.streaming = true
    await settle(view)

    assert.deepEqual(confirmations, [{ message: '删除这条回答？', title: '删除消息' }])
    assert.equal(store.streaming, true)
    assert.equal(callsOf('delete').length, 0, '流式期间不得发出删除请求')
    assert.deepEqual(
      store.messages.map((message) => message.id),
      [1, 2],
      '没有真正删除时消息必须留在本地'
    )
    assert.match(view.text(), /你好，世界/, '没有真正删除时气泡必须留在屏幕上')
    assert.deepEqual(
      view.messages.filter((message) => message.level === 'success'),
      [],
      'store 返回 false 时不得报「已删除」'
    )
    assert.deepEqual(
      view.messages.filter((message) => message.level === 'warning').map((message) => message.message),
      ['正在生成回答，暂不能删除'],
      '应当明确提示此刻不能删除'
    )
  } finally {
    await close(view)
  }
})

test('删除按钮：流式生成期间全部禁用，点击也不发请求', async () => {
  const { view } = await mountChat({ route: '/chat/c1' })

  try {
    // 让流式请求悬着不返回，把 streaming 固定在 true。
    globalThis.fetch = () => new Promise(() => {})
    globalThis.__streamCalls = []
    await sendQuestion(view, '流式中的新问题')
    await view.flush(3)
    assert.equal((await viewStore(view)).streaming, true, '前置：此时应当在流式生成中')

    const buttons = view.queryAll('[data-testid="delete-message"]')
    assert.ok(buttons.length >= 2, '流式期间既有消息的入口仍应渲染')
    for (const button of buttons) {
      assert.equal(button.disabled, true, '流式期间删除入口应当禁用')
    }

    // disabled 只挡用户点击；视图里的 streaming 守卫还要挡住程序化派发。
    deleteButtons(view, 'assistant')[0].dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await view.flush(2)
    assert.equal(confirmations.length, 0, '流式期间不得弹出确认框')
    assert.equal(callsOf('delete').length, 0, '流式期间不得发出删除请求')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// l. 编辑用户消息并重发（issue #253）
//
// 「编辑」和 #256 的「删除」是同一条服务端截断路径：先 `DELETE /chat/messages/{id}`
// 让后端把这条提问及其后的回答删掉，再用新的文本走原来的流式发送。视图不能只改
// 本地 messages —— 否则刷新一次旧回答就会从服务端回来。这一组用例钉住：
//   - 入口只挂在用户消息上，进入编辑后带原文并全选；
//   - 提交恰好一次 DELETE，随后流式请求体的 question 是新文本、附件沿用原消息；
//   - 取消（Esc / 失焦 / 按钮）没有 DELETE、没有流式请求，消息原样保留；
//   - IME 组字中的 Enter 不提交（与主输入框同一条守卫）；
//   - 流式期间入口禁用、提交被拒，取消不发请求；
//   - 重新生成也先 DELETE 原提问，而不是只在前端 replaceMessages。
// ---------------------------------------------------------------------------

const EDIT_ATTACHMENT = { object_key: 'att-1', url: '/uploads/a.png', name: 'a.png' }

const EDIT_MESSAGES = [
  { id: 1, role: 'user', content: '原始问题', attachments: [EDIT_ATTACHMENT] },
  { id: 2, role: 'assistant', content: '旧回答' },
]

// 编辑/重发生效后服务端的落库结果：流结束时的 refreshMessages 会用它重建列表，
// 若继续返回旧的 PERSISTED，旧回答会被这次刷新带回来（用例就会红在错误的地方）。
function postEditMessages(question) {
  return [
    { id: 11, role: 'user', content: question, attachments: [EDIT_ATTACHMENT] },
    { id: 12, role: 'assistant', content: '你好，世界' },
  ]
}

// 传函数而不是数组：用例要在动作命中服务端之后换一份「新的落库结果」，
// 关闭包住数组变量的话，后面再赋值外面那个变量也改不到这里。
function stubChatGets(getServerMessages) {
  respond('get', (url) => {
    if (url === '/knowledge-bases') return [KB1, KB2]
    if (url === '/chat/conversations') return [CONV1]
    if (url === '/chat/conversations/c1') return getServerMessages()
    return {}
  })
}

function editButtons(view) {
  return view.queryAll('[data-testid="edit-message"][data-message-role="user"]')
}

function editTextarea(view) {
  const input = view.query('[data-testid="edit-message-input"]')
  if (!input) return null
  return input.tagName === 'TEXTAREA' ? input : input.querySelector('textarea')
}

async function openEditor(view) {
  const button = editButtons(view)[0]
  assert.ok(button, '应当有用户消息的编辑入口')
  button.click()
  await view.flush(4)
  return editTextarea(view)
}

async function typeEdit(view, text) {
  const textarea = editTextarea(view)
  assert.ok(textarea, '进入编辑后应当出现行内输入框')
  textarea.value = text
  textarea.dispatchEvent(new Event('input', { bubbles: true }))
  await view.flush(3)
  return textarea
}

function pressEditEnter(view, init = {}) {
  const textarea = editTextarea(view)
  assert.ok(textarea, '编辑态应当有 textarea')
  const event = new KeyboardEvent('keydown', {
    key: 'Enter', code: 'Enter', bubbles: true, cancelable: true, ...init,
  })
  textarea.dispatchEvent(event)
  return event
}

test('编辑入口：只挂在用户消息上，进入后输入框带原文并全选', async () => {
  const { view } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })

  try {
    const buttons = editButtons(view)
    assert.equal(buttons.length, 1, 'PERSISTED 里只有一条用户消息，应当只有一个编辑入口')
    assert.equal(buttons[0].disabled, false, '已落库的用户消息应当可编辑')
    assert.equal(view.queryAll('[data-testid="edit-message"][data-message-role="assistant"]').length, 0,
      '助手消息不应当有编辑入口')

    const textarea = await openEditor(view)

    // 承重：编辑框里是原文，而且进入编辑时聚焦 + 全选，用户可以直接覆盖。
    assert.equal(textarea.value, '原始问题', '编辑框应当带出原消息内容')
    assert.equal(document.activeElement, textarea, '进入编辑后应当聚焦输入框')
    assert.equal(textarea.selectionStart, 0, '进入编辑时应当从第一个字符开始全选')
    assert.equal(textarea.selectionEnd, '原始问题'.length, '进入编辑时应当选中全部原文')
    assert.equal(editButtons(view).length, 0, '编辑中不应再显示同一条消息的入口')
    assert.match(view.text(), /保存并重发/, '应当提供提交按钮')
    assert.match(view.text(), /取消/, '应当提供取消按钮')
  } finally {
    await close(view)
  }
})

test('编辑提交：恰好一次 DELETE 截断旧问答，再以新文本重发并沿用附件', async () => {
  const { view, streamCalls } = await mountChat({
    route: '/chat/c1',
    messages: EDIT_MESSAGES,
  })
  globalThis.__streamCalls = streamCalls

  try {
    let serverMessages = EDIT_MESSAGES
    stubChatGets(() => serverMessages)

    await openEditor(view)
    await typeEdit(view, '修改后的问题')

    // 服务端在 DELETE 之后、流结束刷新之前的状态：旧回答已经没了。
    serverMessages = postEditMessages('修改后的问题')

    const event = pressEditEnter(view)
    assert.equal(event.defaultPrevented, true, '非合成态的 Enter 应当提交编辑')

    await until(view, () => streamCalls.length > 0)
    await until(view, () => view.text().includes('你好，世界'))
    await settle(view)

    // 承重①：后端截断路径真的走了，且只走一次。
    // 带 keep_attachments=true：紧接着重发的是同一批 object_key，后端不能把它们登记成墓碑。
    assert.deepEqual(callsOf('delete'), [
      { method: 'delete', args: ['/chat/messages/1', { params: { keep_attachments: true } }] },
    ], '编辑提交应当先删掉原提问及其后的回答，并保留附件登记')
    assert.equal(streamCalls.length, 1, '截断成功后应当恰好重发一次')
    const body = JSON.parse(streamCalls[0].options.body)
    assert.equal(body.question, '修改后的问题', '流式请求体应当是新文本')
    assert.deepEqual(body.attachments, [EDIT_ATTACHMENT], '原消息的附件应当一并带上')

    // 承重②：编辑/重发不经过删除确认框（它是明确的用户动作）。
    assert.deepEqual(confirmations, [], '编辑提交不应弹出删除确认框')

    // 承重③：旧回答和旧提问都从屏幕上消失，编辑态收尾干净。
    assert.doesNotMatch(view.text(), /旧回答/, '旧回答应当随截断从视图消失')
    assert.match(view.text(), /修改后的问题/, '新提问应当上屏')
    assert.equal(view.query('[data-testid="edit-message-editor"]'), null, '提交后应当退出编辑态')
    assert.equal(editTextarea(view), null, '提交后不应残留编辑输入框')

    const store = await viewStore(view)
    assert.deepEqual(store.messages.map((message) => message.id), [11, 12],
      '刷新后应当只剩服务端的「新提问 + 新回答」')
  } finally {
    await close(view)
  }
})

test('编辑提交：内容为空或与原文相同都不删不重发，只退出编辑', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    const store = await viewStore(view)

    for (const draft of ['   ', '原始问题']) {
      await openEditor(view)
      await typeEdit(view, draft)
      const event = pressEditEnter(view)
      assert.equal(event.defaultPrevented, true, '空/未改的 Enter 仍应被编辑框消费（不换行）')
      await settle(view)

      assert.equal(callsOf('delete').length, 0, `「${draft}」不应删除服务端消息`)
      assert.equal(streamCalls.length, 0, `「${draft}」不应发出流式请求`)
      assert.equal(view.query('[data-testid="edit-message-editor"]'), null, '应当直接退出编辑态')
      assert.deepEqual(store.messages.map((message) => message.id), [1, 2], '消息应当原样保留')
    }

    // 退出编辑后原文回到屏幕上，且入口重新可用。
    assert.match(view.text(), /原始问题/)
    assert.equal(editButtons(view)[0].disabled, false)
  } finally {
    await close(view)
  }
})

test('编辑取消：Esc 不发请求、丢弃草稿并还原消息', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    await openEditor(view)
    const textarea = await typeEdit(view, '改到一半')
    const event = new KeyboardEvent('keydown', {
      key: 'Escape', code: 'Escape', bubbles: true, cancelable: true,
    })
    textarea.dispatchEvent(event)
    await settle(view)

    assert.equal(event.defaultPrevented, true, 'Esc 应当被编辑框消费')
    assert.equal(callsOf('delete').length, 0, '取消不得发出删除请求')
    assert.equal(streamCalls.length, 0, '取消不得发出流式请求')
    assert.equal(view.query('[data-testid="edit-message-editor"]'), null, '取消后应当退出编辑态')
    assert.doesNotMatch(view.text(), /改到一半/, '取消后草稿不得留在屏幕上')
    assert.match(view.text(), /原始问题/, '取消后原文应当回到屏幕上')

    const store = await viewStore(view)
    assert.deepEqual(store.messages.map((message) => message.id), [1, 2], '取消后消息不变')
  } finally {
    await close(view)
  }
})

test('编辑取消：失焦不发请求、丢弃草稿并还原消息', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    await openEditor(view)
    const textarea = await typeEdit(view, '失焦时的草稿')
    // startEdit 已经 focus 过；这里再显式聚焦，保证 blur() 真的会派发事件。
    textarea.focus()
    textarea.blur()
    await settle(view)

    assert.equal(callsOf('delete').length, 0, '失焦取消不得发出删除请求')
    assert.equal(streamCalls.length, 0, '失焦取消不得发出流式请求')
    assert.equal(view.query('[data-testid="edit-message-editor"]'), null, '失焦后应当退出编辑态')
    assert.doesNotMatch(view.text(), /失焦时的草稿/, '失焦后草稿不得留在屏幕上')
    assert.match(view.text(), /原始问题/, '失焦后原文应当回到屏幕上')

    const store = await viewStore(view)
    assert.deepEqual(store.messages.map((message) => message.id), [1, 2], '失焦取消后消息不变')
  } finally {
    await close(view)
  }
})

test('编辑输入：IME 组字中 / keyCode 229 与 Shift+Enter 都不提交', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    // isComposing=true（现代引擎）与仅 keyCode=229（旧引擎）都必须让给输入法选词。
    for (const init of [{ isComposing: true }, { keyCode: 229 }]) {
      await openEditor(view)
      await typeEdit(view, '输入法里的草稿')
      const event = pressEditEnter(view, init)
      await settle(view)

      assert.equal(event.defaultPrevented, false, '组字中的 Enter 不得被编辑框吃掉')
      assert.equal(callsOf('delete').length, 0, '组字中的 Enter 不得触发删除')
      assert.equal(streamCalls.length, 0, '组字中的 Enter 不得触发重发')
      assert.ok(editTextarea(view), '组字中的 Enter 后编辑框和草稿应当保留')

      // 退出编辑，给下一轮一个干净的起点。
      view.query('[data-testid="cancel-edit-message"]').dispatchEvent(
        new MouseEvent('click', { bubbles: true })
      )
      await view.flush(2)
    }

    // Shift+Enter 在编辑框里是换行，不是提交。
    await openEditor(view)
    await typeEdit(view, '要换行')
    const event = pressEditEnter(view, { shiftKey: true })
    await settle(view)
    assert.equal(event.defaultPrevented, false, 'Shift+Enter 不得被 preventDefault')
    assert.equal(callsOf('delete').length, 0, 'Shift+Enter 不得触发删除')
    assert.equal(streamCalls.length, 0, 'Shift+Enter 不得触发重发')
    assert.ok(editTextarea(view), 'Shift+Enter 后应当继续编辑')
  } finally {
    await close(view)
  }
})

test('编辑的流式保护：入口禁用、提交被拒，取消不发请求', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    // 先进入编辑、写好草稿，再让会话进入流式（模拟生成过程中用户点了编辑的场景）。
    await openEditor(view)
    await typeEdit(view, '流式中不改了')
    const store = await viewStore(view)
    store.streaming = true
    await view.flush(3)

    // 提交按钮本身不靠 disabled 挡（它只防重复提交），必须由 submitEdit 的 streaming 守卫拦下。
    view.query('[data-testid="submit-edit-message"]').dispatchEvent(
      new MouseEvent('click', { bubbles: true })
    )
    await settle(view)
    assert.equal(callsOf('delete').length, 0, '流式期间提交不得发出删除请求')
    assert.equal(streamCalls.length, 0, '流式期间提交不得发出流式请求')
    assert.ok(editTextarea(view), '提交被拒时编辑框和草稿应当保留')

    // 取消始终可用，而且不产生任何请求。
    view.query('[data-testid="cancel-edit-message"]').dispatchEvent(
      new MouseEvent('click', { bubbles: true })
    )
    await view.flush(2)
    assert.equal(view.query('[data-testid="edit-message-editor"]'), null, '取消后应当退出编辑态')
    assert.equal(callsOf('delete').length, 0, '流式期间取消不得发出删除请求')
    assert.equal(streamCalls.length, 0, '流式期间取消不得发出流式请求')

    // 入口渲染为禁用；绕过 disabled 的程序化点击也进不去编辑态。
    const button = editButtons(view)[0]
    assert.equal(button.disabled, true, '流式期间编辑入口应当禁用')
    button.dispatchEvent(new MouseEvent('click', { bubbles: true }))
    await view.flush(2)
    assert.equal(view.query('[data-testid="edit-message-editor"]'), null,
      '流式期间程序化点击也不得进入编辑态')
  } finally {
    await close(view)
  }
})

test('编辑提交：store 拒绝删除时提示并保留编辑内容，不重发', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    const store = await viewStore(view)
    // store 在流式期间返回 false：视图必须当作「没删成」，不能继续发送。
    store.deleteMessage = async () => false

    await openEditor(view)
    await typeEdit(view, '修改后的问题')
    pressEditEnter(view)
    await settle(view)

    assert.equal(callsOf('delete').length, 0, 'store 未真正删除时不应有请求发出')
    assert.equal(streamCalls.length, 0, '删除失败后不得重发')
    assert.deepEqual(
      view.messages.filter((message) => message.level === 'warning').map((message) => message.message),
      ['正在生成回答，暂不能编辑'],
      '应当明确提示此刻不能编辑'
    )
    assert.ok(editTextarea(view), '失败时编辑框应当保留')
    assert.equal(editTextarea(view).value, '修改后的问题', '失败时草稿应当保留')
    assert.deepEqual(store.messages.map((message) => message.id), [1, 2], '消息应当原样保留')
  } finally {
    await close(view)
  }
})

test('编辑提交：删除请求失败时不重发，编辑框和草稿保留', async () => {
  const { view, streamCalls } = await mountChat({ route: '/chat/c1', messages: EDIT_MESSAGES })
  globalThis.__streamCalls = streamCalls

  try {
    respond('delete', () => {
      throw new Error('删除失败')
    })

    await openEditor(view)
    await typeEdit(view, '修改后的问题')
    pressEditEnter(view)
    await settle(view)

    assert.equal(callsOf('delete').length, 1, '删除请求应当发出一次')
    assert.equal(streamCalls.length, 0, '删除失败后不得重发')
    assert.ok(editTextarea(view), '失败时编辑框应当保留')
    assert.equal(editTextarea(view).value, '修改后的问题', '失败时草稿应当保留')

    const store = await viewStore(view)
    assert.deepEqual(store.messages.map((message) => message.id), [1, 2], '失败时消息不变')
  } finally {
    await close(view)
  }
})

test('重新生成：先 DELETE 原提问做服务端截断，再重发原问题与附件', async () => {
  const { view } = await mountChat({
    route: '/chat/c1',
    messages: EDIT_MESSAGES,
    streamFrames: ANSWER_FRAMES,
  })

  try {
    let serverMessages = EDIT_MESSAGES
    stubChatGets(() => serverMessages)
    // 重新生成的回答必须和旧回答不同，才能断言旧内容确实随截断消失。
    const streamCalls = installStreamStub(ANSWER_FRAMES)
    globalThis.__streamCalls = streamCalls

    serverMessages = postEditMessages('原始问题')

    view.query('[data-testid="regenerate-message"]').click()

    await until(view, () => streamCalls.length > 0)
    await until(view, () => view.text().includes('你好，世界'))
    await settle(view)

    // 承重：走的是后端截断（DELETE 原提问 id=1），不是前端 replaceMessages；
    // 同样带 keep_attachments=true，否则重发时附件键已被回收。
    assert.deepEqual(callsOf('delete'), [
      { method: 'delete', args: ['/chat/messages/1', { params: { keep_attachments: true } }] },
    ], '重新生成应当删除原提问及其后的回答，并保留附件登记')
    assert.deepEqual(confirmations, [], '重新生成不应弹出删除确认框')
    assert.equal(streamCalls.length, 1, '截断成功后应当恰好重发一次')
    const body = JSON.parse(streamCalls[0].options.body)
    assert.equal(body.question, '原始问题', '重发的应当还是原问题')
    assert.deepEqual(body.attachments, [EDIT_ATTACHMENT], '原提问的附件应当一并带上')
    assert.doesNotMatch(view.text(), /旧回答/, '旧回答应当随截断从视图消失')

    const store = await viewStore(view)
    assert.deepEqual(store.messages.map((message) => message.id), [11, 12],
      '刷新后应当只剩「原提问 + 新回答」')
  } finally {
    await close(view)
  }
})
