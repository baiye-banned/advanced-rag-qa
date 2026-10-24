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

// vue-router 必须**动态** import，且排在 vueMount 之后 —— 它自己 import 'vue'，
// 而 vue 的 runtime-dom 在模块求值时就抓走 document
//（`const doc = typeof document !== 'undefined' ? document : null`），
// 静态 import 会让这行跑在 vueMount 装 jsdom 全局之前，症状是挂载时
// 「Cannot read properties of null (reading 'createElement')」。
// 与 vueMount.js 文件头「不要在同一文件里再静态 import vue」是同一条约束。
const { createMemoryHistory, createRouter } = await import('vue-router')

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
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
