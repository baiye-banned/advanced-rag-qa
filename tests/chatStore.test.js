import test from 'node:test'
import assert from 'node:assert/strict'
import { registerHooks } from 'node:module'

// 桩模块：只替换网络层，跑的是真实的 src/stores/chat.js。
const stubSource = [
  'const pending = new Map()',
  // 反馈是独立的出口：getMessages 的 pending 队列由 respond/respondError 驱动，
  // 混在一起会让「消费一条反馈响应」顺手吃掉一条消息响应。
  'const feedbackPending = new Map()',
  'const conversationResolvers = []',
  // 每次会话列表请求的参数按发出顺序留档：用例据此断言 q / 游标真的带上了。
  'export const conversationRequests = []',
  'export const streams = []',
  'export const getMessagesCalls = []',
  'export const getMessagesParams = []',
  'export const chatAPI = {',
  '  getConversations: (params) => new Promise((resolve) => {',
  '    const entry = { params, resolve }',
  '    conversationRequests.push(entry)',
  '    conversationResolvers.push(entry)',
  '  }),',
  '  getMessages: (id, params) => new Promise((resolve, reject) => {',
  '    getMessagesCalls.push(id)',
  '    getMessagesParams.push(params)',
  '    const queue = pending.get(id) || []',
  '    queue.push({ resolve, reject, params })',
  '    pending.set(id, queue)',
  '  }),',
  '  deleteConversation: async () => {},',
  '  renameConversation: async () => {},',
  '  setMessageFeedback: (id, feedback) => new Promise((resolve, reject) => {',
  '    const queue = feedbackPending.get(id) || []',
  '    queue.push({ resolve, reject, feedback })',
  '    feedbackPending.set(id, queue)',
  '  }),',
  '}',
  'export function respondConversations(list) {',
  '  conversationResolvers.splice(0).forEach((entry) => entry.resolve(list))',
  '}',
  // 乱序返回用：只解析第 index 次发出的列表请求，其余保持在途。
  'export function respondConversation(index, list) {',
  '  conversationRequests[index]?.resolve(list)',
  '}',
  // 按后端语义切片：messages 传该会话的完整消息（id 升序），
  // 这里按 limit / before_id 取「游标之前最新的 limit 条」，与真实接口一致，
  // 才能验证前端发出的游标与页大小是否真的对得上。
  'function resolveWithWindow(entry, messages) {',
  '  const params = entry.params || {}',
  '  const limit = params.limit == null ? 50 : params.limit',
  '  let rows = messages',
  '  if (params.before_id != null) rows = rows.filter((item) => item.id < params.before_id)',
  '  entry.resolve(rows.slice(-limit))',
  '}',
  'export function respond(id, messages) {',
  '  const queue = pending.get(id) || []',
  '  const entry = queue.shift()',
  '  pending.set(id, queue)',
  '  if (entry) resolveWithWindow(entry, messages)',
  '}',
  // 乱序返回用：解析该会话最后一个在途请求，模拟「先发出的请求后返回」。
  'export function respondLatest(id, messages) {',
  '  const queue = pending.get(id) || []',
  '  const entry = queue.pop()',
  '  pending.set(id, queue)',
  '  if (entry) resolveWithWindow(entry, messages)',
  '}',
  'export function respondError(id, error) {',
  '  const queue = pending.get(id) || []',
  '  const entry = queue.shift()',
  '  pending.set(id, queue)',
  '  if (entry) entry.reject(error)',
  '}',
  'export function respondFeedback(id) {',
  '  const queue = feedbackPending.get(id) || []',
  '  const entry = queue.shift()',
  '  feedbackPending.set(id, queue)',
  '  if (entry) entry.resolve({})',
  '}',
  'export function respondFeedbackError(id, error) {',
  '  const queue = feedbackPending.get(id) || []',
  '  const entry = queue.shift()',
  '  feedbackPending.set(id, queue)',
  '  if (entry) entry.reject(error)',
  '}',
  // 每条用例开始前清空桩状态：某条用例提前断言失败时可能留下没人消费的 resolver，
  // 若不清理，下一条用例的 respond() 会把它消费掉，导致该用例永远等不到自己的响应。
  'export function resetStub() {',
  '  pending.clear()',
  '  feedbackPending.clear()',
  '  conversationResolvers.length = 0',
  '  conversationRequests.length = 0',
  '}',
  'export function streamChat(options) {',
  '  streams.push(options)',
  '}',
  '',
].join('\n')

export const chatApiStub = `data:text/javascript,${encodeURIComponent(stubSource)}`

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === '@/api/chat') {
      return { url: chatApiStub, shortCircuit: true }
    }
    if (specifier.startsWith('@/')) {
      return {
        url: new URL(`../src/${specifier.slice(2)}.js`, import.meta.url).href,
        shortCircuit: true,
      }
    }
    return nextResolve(specifier, context)
  },
})

// 评测轮询依赖 window.setInterval；这里只记录调用（回调与间隔），不真正起定时器。
// 轮询用例因此可以直接手动触发某一次 tick：pollingStarts.at(-1)()。
const pollingStarts = []
const pollingDelays = []
const pollingStops = []
globalThis.window = {
  setInterval: (fn, delay) => {
    pollingStarts.push(fn)
    pollingDelays.push(delay)
    return pollingStarts.length
  },
  clearInterval: (timer) => {
    pollingStops.push(timer)
  },
}

const { createPinia, setActivePinia } = await import('pinia')
const { useChatStore } = await import('../src/stores/chat.js')
const { respond, respondLatest, respondError, respondFeedback, respondFeedbackError, respondConversations, respondConversation, conversationRequests, resetStub, streams, getMessagesCalls, getMessagesParams } =
  await import(chatApiStub)

const conversation = (id, knowledgeBaseId = null) => ({
  id,
  title: id,
  knowledge_base_id: knowledgeBaseId,
})

const message = (id, content, ragasStatus = '') => ({
  id,
  role: 'assistant',
  content,
  ragas_status: ragasStatus,
})

const contents = (store) => store.messages.map((item) => item.content)

const flush = () => new Promise((resolve) => setTimeout(resolve, 0))

// 让流的收尾（fetchConversations → 可能的消息刷新）跑完，避免用例留下未决 Promise。
async function settleTail(ids = []) {
  respondConversations([])
  await flush()
  ids.forEach((id) => respond(id, []))
  await flush()
}

function createStore(conversations) {
  setActivePinia(createPinia())
  resetStub()
  streams.length = 0
  getMessagesCalls.length = 0
  getMessagesParams.length = 0
  pollingStarts.length = 0
  pollingDelays.length = 0
  pollingStops.length = 0
  const store = useChatStore()
  store.conversations = conversations
  return store
}

test('selectConversation 丢弃先发出但后返回的陈旧响应', async () => {
  const store = createStore([conversation('a'), conversation('b')])

  const requestA = store.selectConversation('a')
  const requestB = store.selectConversation('b')

  respond('b', [message(2, 'B 的回答')])
  await requestB
  respond('a', [message(1, 'A 的回答')])
  await requestA

  assert.equal(store.currentId, 'b')
  assert.deepEqual(contents(store), ['B 的回答'])
})

test('连续快速切换多个会话，乱序返回后仍只保留最后一个会话的消息', async () => {
  const ids = ['a', 'b', 'c', 'd']
  const store = createStore(ids.map((id) => conversation(id)))

  // 依次点击 a→b→c→d，四个请求同时在空中。
  const pending = ids.map((id) => store.selectConversation(id))

  // 返回顺序与点击顺序完全相反，最后返回的是最早发出的那次。
  for (const index of [3, 1, 0, 2]) {
    respond(ids[index], [message(index, `${ids[index]} 的回答`)])
    await pending[index]
  }

  assert.equal(store.currentId, 'd')
  assert.deepEqual(contents(store), ['d 的回答'])
  assert.equal(store.loading, false)
})

test('selectConversation 成功后才提交会话标识与消息', async () => {
  const store = createStore([conversation('a'), conversation('b')])

  const requestA = store.selectConversation('a')
  respond('a', [message(1, 'A 的回答')])
  await requestA
  assert.equal(store.currentId, 'a')

  const requestB = store.selectConversation('b')
  // 请求在途期间：加载目标已是 b，但「已选中」的会话仍是 a，
  // 视图不会出现「标题换了、内容没换」的中间态。
  assert.equal(store.loadingConversationId, 'b')
  assert.equal(store.currentId, 'a')
  assert.deepEqual(contents(store), ['A 的回答'])

  respond('b', [message(2, 'B 的回答')])
  await requestB

  assert.equal(store.currentId, 'b')
  assert.deepEqual(contents(store), ['B 的回答'])
  assert.equal(store.loadingConversationId, null)
  assert.equal(store.loading, false)
})

test('selectConversation 失败时保留原会话，并让 Promise 拒绝', async () => {
  const store = createStore([conversation('a'), conversation('b')])

  const requestA = store.selectConversation('a')
  respond('a', [message(1, 'A 的回答')])
  await requestA

  const requestB = store.selectConversation('b')
  respondError('b', new Error('network down'))
  await assert.rejects(requestB, /network down/)

  // 失败不得留下「标识是 b、内容是 a」的错位状态：两者都还是 a。
  assert.equal(store.currentId, 'a')
  assert.deepEqual(contents(store), ['A 的回答'])
  assert.equal(store.loadingConversationId, null)
  assert.equal(store.loading, false)
})

test('旧加载晚失败时不影响已经接管的新会话', async () => {
  const store = createStore([conversation('a'), conversation('b'), conversation('c')])

  const requestA = store.selectConversation('a')
  respond('a', [message(1, 'A 的回答')])
  await requestA

  const requestB = store.selectConversation('b')
  const requestC = store.selectConversation('c')
  respondError('b', new Error('network down'))
  await assert.rejects(requestB, /network down/)

  // 最新一次加载（c）还没回来：只丢弃 b，不能把 loading 提前复位。
  assert.equal(store.currentId, 'a')
  assert.deepEqual(contents(store), ['A 的回答'])
  assert.equal(store.loadingConversationId, 'c')
  assert.equal(store.loading, true)

  respond('c', [message(3, 'C 的回答')])
  await requestC

  assert.equal(store.currentId, 'c')
  assert.deepEqual(contents(store), ['C 的回答'])
  assert.equal(store.loading, false)
})

test('selectConversation 的 loading 只由最新一次加载复位', async () => {
  const store = createStore([conversation('a'), conversation('b')])

  const requestA = store.selectConversation('a')
  const requestB = store.selectConversation('b')

  respond('a', [message(1, 'A 的回答')])
  await requestA
  assert.equal(store.loading, true)

  respond('b', [])
  await requestB
  assert.equal(store.loading, false)
})

test('陈旧响应不改动 selectedKnowledgeBaseId，也不启动评测轮询', async () => {
  const store = createStore([
    conversation('a', 'kb-a'),
    conversation('b'),
  ])

  const requestA = store.selectConversation('a')
  const requestB = store.selectConversation('b')

  respond('b', [])
  await requestB
  const pollingBefore = pollingStarts.length

  respond('a', [message(1, 'A 的回答', 'pending')])
  await requestA

  assert.equal(store.currentId, 'b')
  assert.deepEqual(contents(store), [])
  assert.equal(store.selectedKnowledgeBaseId, null)
  assert.equal(pollingStarts.length, pollingBefore)
})

test('最新响应带待评测消息时仍会启动评测轮询', async () => {
  const store = createStore([conversation('a'), conversation('b')])

  const requestB = store.selectConversation('b')
  respond('b', [message(2, 'B 的回答', 'pending')])
  await requestB

  assert.equal(pollingStarts.length, 1)
})

test('加载在飞时清空会话，返回后不得把旧消息写回当前视图', async () => {
  const store = createStore([conversation('a')])

  const requestA = store.selectConversation('a')
  store.clearMessages() // 例如用户在加载期间点了「新对话」
  respond('a', [message(1, 'A 的回答')])
  await requestA

  assert.equal(store.currentId, null)
  assert.deepEqual(contents(store), [])
  assert.equal(store.loading, false)
})

test('加载在飞时流认领了新会话，迟到的加载不得覆盖它和刚发出的提问', async () => {
  const store = createStore([conversation('a')])
  assert.equal(store.currentId, null)

  // 深链进入 /chat/a：消息加载在飞，currentId 仍为空。
  const requestA = store.selectConversation('a')
  assert.equal(store.loadingConversationId, 'a')
  assert.equal(store.currentId, null)

  // 加载还没返回，用户就发起了提问：流为新会话 c9 投递 conversation 事件并认领它。
  store.sendMessage('在途时提问')
  const stream = streams.at(-1)
  stream.onMessage('', { type: 'conversation', conversation: conversation('c9') })
  assert.equal(store.currentId, 'c9')

  // 旧加载这时才返回：只比 seq 的实现会把 currentId 改回 a，并用 a 的历史覆盖新提问。
  respond('a', [message(1, 'A 的历史回答')])
  await requestA

  assert.equal(store.currentId, 'c9')
  assert.deepEqual(contents(store), ['在途时提问'])
  assert.equal(store.loading, false)
  assert.equal(store.loadingConversationId, null)
})

test('被新流式请求取代的旧回调不再写入状态', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('第一个问题')
  const firstStream = streams.at(-1)
  firstStream.onMessage('第一段增量')
  assert.equal(store.streamContent, '第一段增量')

  firstStream.onDone()
  await flush()
  assert.equal(store.streaming, false)

  store.sendMessage('第二个问题')
  const secondStream = streams.at(-1)

  firstStream.onMessage('迟到的增量')
  firstStream.onDone()
  firstStream.onError(new Error('迟到的错误'))

  assert.equal(store.streamContent, '')
  assert.equal(store.streaming, true)

  secondStream.onMessage('第二段增量')
  assert.equal(store.streamContent, '第二段增量')

  await settleTail(['a'])
})

test('切换会话后，旧流的收尾不会写入当前会话的消息', async () => {
  const store = createStore([conversation('a'), conversation('b')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('A 的回答')

  const requestB = store.selectConversation('b')
  respond('b', [message(2, 'B 的回答')])
  await requestB

  stream.onDone()
  respondConversations([conversation('a'), conversation('b')])
  await flush()

  assert.equal(store.currentId, 'b')
  assert.deepEqual(contents(store), ['B 的回答'])
})

test('当前流式请求的错误回调仍然正常收尾', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  streams.at(-1).onError(new Error('模型请求失败'))

  assert.equal(store.streaming, false)
  assert.equal(store.errorMessage, '模型请求失败')
  assert.deepEqual(contents(store), ['问题', '模型请求失败'])
})

test('回答刚结束又立刻追问时，仍在当前会话的收尾仍要刷新消息并启动评测轮询', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('第一个问题')
  const firstStream = streams.at(-1)
  firstStream.onMessage('', { type: 'trace', trace_id: 't1', event: { index: 0, stage: 'retrieval_completed' } })
  firstStream.onMessage('第一段回答')
  firstStream.onDone() // 同步部分已收尾，随后 await 会话列表
  assert.equal(store.streaming, false)

  // 收尾还挂在 fetchConversations() 上时用户继续追问（streaming 已复位，允许发送）。
  store.sendMessage('第二个问题')
  const secondStream = streams.at(-1)

  respondConversations([])
  await flush()
  assert.deepEqual(getMessagesCalls, ['a']) // 收尾没有被新请求作废，仍为当前会话刷新

  respond('a', [message(9, '第一段回答', 'pending')])
  await flush()

  assert.equal(store.currentId, 'a')
  assert.equal(pollingStarts.length, 1) // 刚生成的回答仍会进入评测轮询

  secondStream.onMessage('第二段回答')
  assert.equal(store.streamContent, '第二段回答')
})

test('收尾刷新期间切走再返回，不得改动已切走会话的评测轮询', async () => {
  const store = createStore([conversation('a'), conversation('b')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('回答')
  stream.onDone()

  respondConversations([])
  await flush()
  assert.deepEqual(getMessagesCalls, ['a']) // 收尾已进入 refreshMessages('a')，尚未返回

  // 用户在收尾刷新在飞时切到 B，B 有带评测消息 → 启动轮询。
  const requestB = store.selectConversation('b')
  respond('b', [message(2, 'B 的回答', 'pending')])
  await requestB
  assert.equal(pollingStarts.length, 1)

  // 收尾的旧刷新这时才返回：它属于已切走的会话 a，不得停掉 B 的轮询。
  respond('a', [message(1, 'A 的回答')])
  await flush()

  assert.equal(store.currentId, 'b')
  assert.equal(pollingStops.length, 0)
  assert.deepEqual(contents(store), ['B 的回答'])
})

test('收尾刷新在飞时又追问，旧刷新不得抹掉新提问', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('第一个问题')
  const firstStream = streams.at(-1)
  firstStream.onMessage('第一段回答')
  firstStream.onDone() // 收尾挂在 fetchConversations 上

  respondConversations([])
  await flush()
  assert.deepEqual(getMessagesCalls, ['a']) // 收尾已进入 refreshMessages('a')，尚未返回

  // 旧刷新还没回来，用户又提了第二个问题：新提问的乐观消息已进入列表。
  store.sendMessage('第二个问题')
  const secondStream = streams.at(-1)
  assert.deepEqual(contents(store), ['第一个问题', '第一段回答', '第二个问题'])

  // 收尾的旧刷新这时才返回：它属于上一个请求，不得用后端快照覆盖掉在飞的新提问。
  respond('a', [message(9, '第一段回答', 'pending')])
  await flush()

  assert.deepEqual(contents(store), ['第一个问题', '第一段回答', '第二个问题'])

  secondStream.onMessage('第二段回答')
  assert.equal(store.streamContent, '第二段回答')
})

test('切走后，旧流的会话事件不得再改写当前会话的知识库选择', async () => {
  const store = createStore([conversation('a', 'kb-a'), conversation('b', 'kb-b')])
  store.setCurrentId('a')
  store.setSelectedKnowledgeBaseId('kb-a')

  store.sendMessage('问题')
  const stream = streams.at(-1)

  // 回答生成期间用户切到会话 B：知识库选择应随 B 变成 kb-b。
  const requestB = store.selectConversation('b')
  respond('b', [])
  await requestB
  assert.equal(store.selectedKnowledgeBaseId, 'kb-b')

  // 旧流此刻才投递 conversation 事件（带着 A 的知识库），不得把当前会话的选择改回 kb-a。
  stream.onMessage('', { type: 'conversation', conversation: conversation('a', 'kb-a') })

  assert.equal(store.currentId, 'b')
  assert.equal(store.selectedKnowledgeBaseId, 'kb-b')
})

test('流式期间点了「新对话」，旧流不得把会话认领回来', async () => {
  const store = createStore([conversation('a', 'kb-a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('回答')
  stream.onDone() // 收尾挂在 fetchConversations 上

  // 回答还在收尾时用户点了「新对话」：视图被显式清空。
  store.clearMessages()
  assert.equal(store.currentId, null)

  // 旧流此刻才投递 conversation 事件与收尾结果，都不得把已清空的视图认领回会话 a。
  stream.onMessage('', { type: 'conversation', conversation: conversation('a', 'kb-a') })
  respondConversations([conversation('a', 'kb-a')])
  await flush()

  assert.equal(store.currentId, null)
  assert.equal(store.selectedKnowledgeBaseId, null)
  // 待跳转会话也不得登记：Chat.vue 的 watcher 会据此 router.replace('/chat/a')，把用户拽回旧会话。
  assert.equal(store.pendingRouteConversationId, null)
  assert.deepEqual(contents(store), [])
})

test('收尾刷新失败时，仍要启动评测轮询而不是中断收尾', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('', { type: 'trace', trace_id: 't1', event: { index: 0, stage: 'retrieval_completed' } })
  stream.onMessage('回答')
  stream.onDone()

  respondConversations([])
  await flush()
  assert.deepEqual(getMessagesCalls, ['a']) // 收尾已进入 refreshMessages('a')

  // 收尾的 GET /messages 失败（500 / 超时）：不得把异常抛进 onDone（streamChat 丢弃该 Promise，
  // 会变成未处理的 rejection），待评测的本地消息仍要进入轮询，靠轮询重试收敛。
  respondError('a', new Error('消息接口失败'))
  await flush()

  assert.equal(pollingStarts.length, 1)
  assert.deepEqual(contents(store), ['问题', '回答'])
})

test('流式期间点了「新对话」，迟到的会话事件不得登记待跳转会话', async () => {
  const store = createStore([conversation('a', 'kb-a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  // 回答还在生成，用户点了「新对话」：视图被清空，路由回到 /chat。
  store.clearMessages()

  // 旧流此刻才投递 conversation 事件，带着会话 a。除了不得认领会话，也不得写入待跳转会话，
  // 否则 Chat.vue 的 watcher 会 router.replace('/chat/a')，再由路由 watcher 把用户拽回旧会话。
  stream.onMessage('', { type: 'conversation', conversation: conversation('a', 'kb-a') })

  assert.equal(store.pendingRouteConversationId, null)
  assert.equal(store.currentId, null)

  stream.onMessage('回答')
  stream.onDone()
  respondConversations([conversation('a', 'kb-a')])
  await flush()
  assert.equal(store.pendingRouteConversationId, null)
  assert.deepEqual(contents(store), [])
})

// ---------------------------------------------------------------------------
// issue #262：清空会话（新建对话 / 登出）必须中止在途流。
// 只中止而不作废回调，或只清视图而不复位 streaming，都会让新对话的首条消息发不出去，
// 或者让旧流把迟到的增量 / 停止标记写回已清空的视图。
// ---------------------------------------------------------------------------

const abortError = () => Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' })

test('清空会话会中止在途流并立即复位流状态', () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('回答了一半')
  assert.equal(store.streaming, true)

  store.clearMessages()

  assert.equal(store.streaming, false)
  assert.equal(store.currentId, null)
  assert.deepEqual(store.messages, [])
  assert.equal(store.hasMoreMessages, false)
  assert.equal(store.streamContent, '')
  assert.equal(store.streamingConversationId, null)
  assert.equal(stream.signal.aborted, true)
})

test('清空会话后可以立即发起新提问，旧流回调不再写入新视图', () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('旧问题')
  const oldStream = streams.at(-1)
  oldStream.onMessage('旧回答')
  store.clearMessages()

  const streamsBefore = streams.length
  store.sendMessage('新问题')

  assert.equal(streams.length, streamsBefore + 1) // 不被旧的 streaming 状态挡住
  const newStream = streams.at(-1)
  assert.notEqual(newStream, oldStream)
  assert.equal(store.streaming, true)
  assert.deepEqual(contents(store), ['新问题'])

  // 旧流迟到的回调（含 abort 收尾）全部失效，既不改内容也不补停止标记。
  oldStream.onMessage('迟到的增量')
  oldStream.onError(abortError())
  oldStream.onDone()
  assert.equal(store.streamContent, '')
  assert.deepEqual(contents(store), ['新问题'])
  assert.equal(store.streaming, true)

  newStream.onMessage('新回答')
  assert.equal(store.streamContent, '新回答')

  store.clearMessages() // 收尾，避免留下在途流
})

test('手动停止生成仍然保留已生成内容与停止标记', () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('半截回答')

  store.stopGeneration()
  stream.onError(abortError())

  assert.equal(stream.signal.aborted, true)
  assert.equal(store.streaming, false)
  assert.deepEqual(contents(store), ['问题', '半截回答\n\n*(已停止生成)*'])
})

test('删除当前正在流式回答的会话会中止流并清空视图', async () => {
  const store = createStore([conversation('a'), conversation('b')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('回答了一半')

  await store.deleteConversation('a')

  assert.equal(stream.signal.aborted, true)
  assert.equal(store.streaming, false)
  assert.equal(store.currentId, null)
  assert.deepEqual(store.messages, [])

  // 旧流收尾不得把已删除会话认领回来。
  stream.onError(abortError())
  stream.onDone()
  assert.equal(store.currentId, null)
  assert.deepEqual(contents(store), [])
})

test('后备模型接管时，reset 事件清空已渲染内容，只保留重置后的回答', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('迟到30分钟以内怎么罚款？')
  const stream = streams.at(-1)

  // 首个模型（DeepSeek）中途断开，这两段增量已经渲染出来了。
  stream.onMessage('根据《员工手册》考勤管理')
  stream.onMessage('，迟到30分钟以内')
  const placeholderMessages = store.messages.length

  // 后端在切到后备模型之前下发 reset：作废已渲染内容，但消息占位（流式气泡）必须保留。
  stream.onMessage('', { type: 'reset', reason: 'text_fallback', message: '已切换到后备模型重新生成' })

  assert.equal(store.streamContent, '')
  assert.equal(store.streaming, true)
  assert.equal(store.messages.length, placeholderMessages)
  assert.deepEqual(contents(store), ['迟到30分钟以内怎么罚款？'])

  // 后备模型从头输出的内容按正常增量追加。
  stream.onMessage('根据《员工手册》考勤管理章节，')
  stream.onMessage('迟到30分钟以内罚款50元。')
  assert.equal(store.streamContent, '根据《员工手册》考勤管理章节，迟到30分钟以内罚款50元。')

  // 本地消息由 streamContent 构建（handleStreamDone 同步部分），必须只含后备模型的回答。
  stream.onDone()
  assert.deepEqual(contents(store), [
    '迟到30分钟以内怎么罚款？',
    '根据《员工手册》考勤管理章节，迟到30分钟以内罚款50元。',
  ])

  await settleTail(['a'])
})

test('后备模型也失败时，reset 已作废的半截回答不得混进错误消息', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('半截回答')
  stream.onMessage('', { type: 'reset', reason: 'text_fallback' })
  stream.onError(new Error('回答生成失败：模型不可用'))

  assert.equal(store.errorMessage, '回答生成失败：模型不可用')
  assert.deepEqual(contents(store), ['问题', '回答生成失败：模型不可用'])
})

test('被新流取代的旧流投递的 reset 不得清空当前流的内容', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')

  store.sendMessage('第一个问题')
  const firstStream = streams.at(-1)
  firstStream.onDone()
  await flush()

  store.sendMessage('第二个问题')
  const secondStream = streams.at(-1)
  secondStream.onMessage('第二个回答')

  // 旧流此刻才投递 reset（#25 的世代守卫必须先拦住它）。
  firstStream.onMessage('', { type: 'reset', reason: 'text_fallback' })

  assert.equal(store.streamContent, '第二个回答')

  secondStream.onDone()
  await settleTail(['a'])
})

// ---------------------------------------------------------------------------
// 评测轮询（startEvaluationPolling / stopEvaluationPolling / markLocalEvaluationTimeout）
// 与 refreshMessages 的消息合并。
//
// 这些函数没有从 store 导出，用例只断言可观测行为：window.setInterval / clearInterval
// 的调用（记录型桩，全程零真实定时器）、chatAPI.getMessages 的调用次数与目标会话、
// 以及 store.messages 的内容与评测状态。
// ---------------------------------------------------------------------------

const backendMessage = (id, role, content, ragasStatus = '') => ({
  id,
  role,
  content,
  ragas_status: ragasStatus,
})

// 手动驱动一次轮询 tick：定时器桩只登记回调，这里显式调用它。
// getMessages 的订阅在回调同步段内完成，因此先触发 tick 再投递响应即可。
async function tick(conversationId, snapshot) {
  const pending = pollingStarts.at(-1)()
  if (snapshot) respond(conversationId, snapshot)
  return pending
}

// 走真实链路进入轮询状态：提问 → 流式回答（带检索轨迹，才会标记待评测）→ 收尾刷新。
// tailSnapshot 是收尾那次 GET /messages 的后端快照。
async function enterPolling(conversationId, tailSnapshot) {
  const store = createStore([conversation(conversationId)])
  store.setCurrentId(conversationId)
  store.sendMessage('问题')
  const stream = streams.at(-1)
  stream.onMessage('', {
    type: 'trace',
    trace_id: 't1',
    event: { index: 0, stage: 'retrieval_completed' },
  })
  stream.onMessage('回答')
  stream.onDone()
  respondConversations([])
  await flush()
  respond(conversationId, tailSnapshot)
  await flush()
  return store
}

test('评测轮询：后端状态收敛为已完成时停止轮询', async () => {
  const store = await enterPolling('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'pending'),
  ])

  assert.equal(pollingStarts.length, 1)
  assert.equal(pollingDelays.at(-1), 3000) // 固定 3 秒一次
  assert.equal(pollingStops.length, 0)

  await tick('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'done'),
  ])

  assert.deepEqual(getMessagesCalls, ['a', 'a']) // 收尾一次 + 每个 tick 一次
  assert.equal(pollingStops.length, 1)
  assert.equal(pollingStops.at(-1), 1) // 停的是自己那块表
  assert.deepEqual(contents(store), ['问题', '回答'])
  assert.equal(store.messages.at(-1).ragas_status, 'done')
  assert.equal(store.messages.at(-1).isLocal, false) // 以服务端快照为准
})

test('评测轮询：后端状态收敛为失败时同样停止轮询', async () => {
  const store = await enterPolling('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'pending'),
  ])

  await tick('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'failed'),
  ])

  assert.equal(pollingStops.length, 1)
  assert.equal(store.messages.at(-1).ragas_status, 'failed')
  assert.deepEqual(contents(store), ['问题', '回答'])
})

test('评测轮询：仍为等待中/评测中时按间隔继续轮询，且不重复起表', async () => {
  // 只关心起表次数，不读 store：不绑定返回值，免得留一个没人用的变量。
  await enterPolling('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'pending'),
  ])

  await tick('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'running'),
  ])
  await tick('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'pending'),
  ])

  assert.equal(pollingStops.length, 0)
  assert.equal(pollingStarts.length, 1) // 复用同一块表，不叠加定时器
  assert.deepEqual(getMessagesCalls, ['a', 'a', 'a'])
})

test('评测轮询：某次刷新失败不中断轮询', async () => {
  const store = await enterPolling('a', [backendMessage(1, 'user', '问题')])

  const pending = pollingStarts.at(-1)()
  respondError('a', new Error('消息接口失败'))
  await pending

  assert.equal(pollingStops.length, 0)
  assert.equal(store.messages.at(-1).ragas_status, 'pending') // 本地待评测消息仍在，靠下次 tick 收敛
})

test('评测轮询：会话切走后下一次 tick 自行停止，且不再请求旧会话', async () => {
  const store = await enterPolling('a', [backendMessage(1, 'user', '问题')])
  const callsBefore = getMessagesCalls.length
  const contentsBefore = contents(store)

  // 用户切到别的会话（不等轮询回调），旧表留到下一次 tick 才被回调自行停掉。
  store.setCurrentId('b')
  await pollingStarts.at(-1)()

  assert.equal(getMessagesCalls.length, callsBefore) // 不为已切走的会话发请求
  assert.equal(pollingStops.at(-1), 1)
  assert.deepEqual(contents(store), contentsBefore) // 也不写旧会话的消息
})

test('评测轮询：超过时限后把本地待评测消息标记为失败并停止轮询', async () => {
  // 后端快照里还没有助手回答：本地乐观消息必须保留，并一直轮询到超时。
  const store = await enterPolling('a', [backendMessage(1, 'user', '问题')])

  assert.equal(pollingStarts.length, 1)
  assert.equal(pollingDelays.at(-1), 3000)
  assert.equal(store.messages.at(-1).isLocal, true)
  assert.equal(store.messages.at(-1).ragas_status, 'pending')

  const snapshot = [backendMessage(1, 'user', '问题')]
  // 时限 190000ms、每次 tick 记 3000ms：第 63 次仍是 189000ms，尚未超时。
  for (let i = 0; i < 63; i += 1) {
    await tick('a', snapshot)
  }
  assert.equal(store.messages.at(-1).ragas_status, 'pending')
  assert.equal(pollingStops.length, 0)

  // 第 64 次 tick（192000ms）越界：标记本地消息并停表。
  await tick('a', snapshot)

  assert.equal(pollingStops.length, 1)
  assert.equal(pollingStarts.length, 1)
  assert.equal(getMessagesCalls.length, 65) // 收尾一次 + 64 次 tick
  const timedOut = store.messages.at(-1)
  assert.equal(timedOut.isLocal, true)
  assert.equal(timedOut.ragas_status, 'failed')
  assert.equal(timedOut.ragas_error, '评测未及时返回，稍后刷新会话可查看最终状态')
  assert.deepEqual(contents(store), ['问题', '回答'])
})

test('评测轮询：只有待评测的助手消息才启动轮询', async () => {
  const store = createStore([
    conversation('a'),
    conversation('b'),
    conversation('c'),
    conversation('d'),
  ])

  // 评测中（running）也算待评测。
  const requestA = store.selectConversation('a')
  respond('a', [backendMessage(2, 'assistant', '回答', 'running')])
  await requestA
  assert.equal(pollingStarts.length, 1)

  // 切到已完成的会话：不新起轮询，并且停掉上一个会话的表。
  const requestB = store.selectConversation('b')
  respond('b', [backendMessage(3, 'assistant', '回答', 'done')])
  await requestB
  assert.equal(pollingStarts.length, 1)
  assert.deepEqual(pollingStops, [1])

  // 助手消息未带评测状态（历史消息）：不启动。
  const requestC = store.selectConversation('c')
  respond('c', [backendMessage(4, 'assistant', '历史回答', '')])
  await requestC
  assert.equal(pollingStarts.length, 1)
  assert.deepEqual(pollingStops, [1])

  // 只有用户提问、还没有回答：不启动。
  const requestD = store.selectConversation('d')
  respond('d', [backendMessage(5, 'user', '只有提问')])
  await requestD
  assert.equal(pollingStarts.length, 1)
  assert.deepEqual(pollingStops, [1])
})

test('消息合并：后端已有同内容的助手回答时，不重复插入本地待评测消息', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')
  store.addMessage({ role: 'user', content: '问题' })
  store.addMessage({ role: 'assistant', content: '回答', ragas_status: 'pending', isLocal: true })

  const request = store.refreshMessages('a')
  respond('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '回答', 'done'),
  ])
  const merged = await request

  assert.equal(merged.length, 2) // 不出现第三条重复回答
  assert.deepEqual(contents(store), ['问题', '回答'])
  assert.equal(store.messages.at(-1).isLocal, false)
  assert.equal(store.messages.at(-1).ragas_status, 'done')
})

test('消息合并：后端快照还没有该回答时保留本地待评测消息', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')
  store.addMessage({ role: 'user', content: '问题' })
  store.addMessage({ role: 'assistant', content: '回答', ragas_status: 'pending', isLocal: true })

  const request = store.refreshMessages('a')
  respond('a', [backendMessage(1, 'user', '问题')])
  const merged = await request

  assert.equal(merged.length, 2)
  assert.equal(store.messages.at(-1).isLocal, true)
  assert.equal(store.messages.at(-1).ragas_status, 'pending')
})

test('消息合并：后端已有助手回答时，本地待评测消息不再插入', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')
  store.addMessage({ role: 'user', content: '问题' })
  store.addMessage({ role: 'assistant', content: '回答', ragas_status: 'pending', isLocal: true })

  const request = store.refreshMessages('a')
  respond('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '另一个答案', 'pending'),
  ])
  await request

  // 后端已落库（助手消息存在），本地这条 pending 交给后端快照接管，避免出现两条回答。
  assert.deepEqual(contents(store), ['问题', '另一个答案'])
  assert.equal(store.messages.at(-1).isLocal, false)
})

test('消息合并：本地已中止的助手消息（非待评测）不与后端快照去重', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')
  store.addMessage({ role: 'user', content: '问题' })
  // 例如用户中断生成留下的本地消息：后端不会落库，只按「内容是否重复」去重。
  store.addMessage({ role: 'assistant', content: '回答\n\n*(已停止生成)*', isLocal: true })

  const request = store.refreshMessages('a')
  respond('a', [
    backendMessage(1, 'user', '问题'),
    backendMessage(2, 'assistant', '另一个答案', 'done'),
  ])
  await request

  assert.deepEqual(contents(store), ['问题', '另一个答案', '回答\n\n*(已停止生成)*'])
})

test('消息合并：会话已切走时只返回合并结果，不写当前视图', async () => {
  const store = createStore([conversation('a'), conversation('b')])
  store.setCurrentId('a')
  store.addMessage({ role: 'user', content: 'A 的问题' })

  const request = store.refreshMessages('a')
  respond('a', [
    backendMessage(1, 'user', 'A 的问题'),
    backendMessage(2, 'assistant', 'A 的回答'),
  ])
  store.setCurrentId('b')
  const merged = await request

  assert.equal(merged.length, 2) // 合并结果照常返回
  assert.deepEqual(contents(store), ['A 的问题']) // 但不改写已切走会话的视图
})


// ---------------------------------------------------------------------------
// 消息级反馈（issue #250）：setMessageFeedback 的乐观更新与失败回滚。
// 「再点同向即取消」这层语义在 Chat.vue 的 onFeedback 里（视图组用例钉住），
// 这里只保证 store 把值落对、失败时回到调用前的值并把异常抛给调用方。
// ---------------------------------------------------------------------------

test('setMessageFeedback：先乐观落态，失败时回滚到调用前的值', async () => {
  const store = createStore([conversation('a')])
  store.setCurrentId('a')
  store.addMessage({ id: 'm1', role: 'assistant', content: '回答' })
  const feedbackOf = () => store.messages.find((item) => item.id === 'm1').feedback

  assert.equal(feedbackOf(), 0)

  // 成功路径：请求还没回来，界面已切到选中态；回来后仍是选中态。
  const liked = store.setMessageFeedback('m1', 1)
  assert.equal(feedbackOf(), 1)
  respondFeedback('m1')
  await liked
  assert.equal(feedbackOf(), 1)

  // 失败路径：回滚到「调用前」的值（这里是 1，不是取反后的 0），并照常抛出。
  const rejected = store.setMessageFeedback('m1', -1)
  assert.equal(feedbackOf(), -1)
  respondFeedbackError('m1', new Error('反馈提交失败'))
  await assert.rejects(rejected, /反馈提交失败/)
  assert.equal(feedbackOf(), 1)
})


// ---- 会话消息分页（issue #61）：后端默认只返回一页，前端按需向前翻 ----
// 桩里的 respond(id, messages) 传的是该会话的完整消息，按后端 limit/before_id 语义切片，
// 因此这里的断言能真正约束「游标对不对、页大小对不对」，而不是只断言调用了接口。

const PAGE = 50
const FETCH = PAGE + 1 // 前端多取一条用于判断还有没有更早的消息

const history = (count) =>
  Array.from({ length: count }, (_, index) => message(index + 1, `第 ${index + 1} 条`))
const ids = (store) => store.messages.map((item) => item.id)

test('selectConversation 只请求最新一页并标记还有更早消息', async () => {
  const store = createStore([conversation('a')])

  const request = store.selectConversation('a')
  respond('a', history(120))
  await request

  assert.deepEqual(getMessagesParams, [{ limit: FETCH }])
  assert.deepEqual(ids(store), ids(store).slice(0, PAGE))
  assert.equal(store.messages.length, PAGE)
  assert.deepEqual(ids(store)[0], 71) // 最新一页：71..120
  assert.equal(store.hasMoreMessages, true)
})

test('消息刚好等于一页时不再提示还有更早消息', async () => {
  const store = createStore([conversation('a')])

  const request = store.selectConversation('a')
  respond('a', history(PAGE))
  await request

  assert.equal(store.messages.length, PAGE)
  assert.equal(store.hasMoreMessages, false) // 51 条才说明还有更早的
})

test('loadOlderMessages 以最旧一条为游标向前翻页，翻完不重复不遗漏', async () => {
  const store = createStore([conversation('a')])
  const all = history(120)
  const open = store.selectConversation('a')
  respond('a', all)
  await open

  const first = store.loadOlderMessages()
  respond('a', all)
  await first

  assert.deepEqual(getMessagesParams.at(-1), { limit: FETCH, before_id: 71 })
  assert.deepEqual(ids(store), Array.from({ length: 100 }, (_, index) => index + 21))

  const second = store.loadOlderMessages()
  respond('a', all)
  await second

  assert.deepEqual(getMessagesParams.at(-1), { limit: FETCH, before_id: 21 })
  assert.deepEqual(ids(store), Array.from({ length: 120 }, (_, index) => index + 1)) // 1..120 各一次
  assert.equal(store.hasMoreMessages, false)
})

test('loadOlderMessages 取满一页时仍可继续向前翻', async () => {
  const store = createStore([conversation('a')])
  const all = history(200)
  const open = store.selectConversation('a')
  respond('a', all)
  await open

  const loading = store.loadOlderMessages()
  respond('a', all)
  await loading

  assert.equal(store.messages.length, 100)
  assert.equal(store.hasMoreMessages, true)
})

test('refreshMessages 不截断已翻出的更早历史，也不凭空点亮「还有更早」', async () => {
  const store = createStore([conversation('a')])
  const all = history(120)
  const open = store.selectConversation('a')
  respond('a', all)
  await open
  const loading = store.loadOlderMessages()
  respond('a', all)
  await loading
  assert.deepEqual(ids(store), Array.from({ length: 100 }, (_, index) => index + 21))

  const refreshing = store.refreshMessages('a')
  respond('a', all)
  await refreshing

  assert.deepEqual(getMessagesParams.at(-1), { limit: FETCH })
  assert.deepEqual(ids(store), Array.from({ length: 100 }, (_, index) => index + 21))
  assert.equal(store.hasMoreMessages, true) // 仍停在第 21 条之前，可以继续往前翻
})

test('整段历史都已加载后，刷新不会重新点亮「加载更早的消息」', async () => {
  const store = createStore([conversation('a')])
  const all = history(120)
  const open = store.selectConversation('a')
  respond('a', all)
  await open
  for (let i = 0; i < 3; i += 1) {
    const loading = store.loadOlderMessages()
    respond('a', all)
    await loading
  }
  assert.deepEqual(ids(store), Array.from({ length: 120 }, (_, index) => index + 1))
  assert.equal(store.hasMoreMessages, false)

  const refreshing = store.refreshMessages('a')
  respond('a', all)
  await refreshing

  assert.equal(store.messages.length, 120)
  assert.equal(store.hasMoreMessages, false) // 没有更早的消息了，按钮不该再出现
})

test('重新生成截断视图后，刷新重建窗口而不是在中间留空洞', async () => {
  const store = createStore([conversation('a')])
  const all = history(120)
  const open = store.selectConversation('a')
  respond('a', all)
  await open
  for (let round = 0; round < 2; round += 1) {
    const loading = store.loadOlderMessages()
    respond('a', all)
    await loading
  }
  assert.deepEqual(ids(store), Array.from({ length: 120 }, (_, index) => index + 1))

  // 重新生成：砍掉后半段，只留第 1 条（Chat.vue 的 regenerate 就是这么调用的）
  store.replaceMessages(store.messages.slice(0, 1))
  assert.deepEqual(ids(store), [1])

  const refreshing = store.refreshMessages('a')
  respond('a', all)
  await refreshing

  // 窗口重建为「最新一页」，而不是「[1] + 最新一页」这种中间断档、翻不回去的拼接
  assert.deepEqual(ids(store), Array.from({ length: 50 }, (_, index) => index + 71))
  assert.equal(store.hasMoreMessages, true)

  // 从这个窗口继续往前翻，整段历史依然一条不少地可达
  for (let round = 0; round < 2; round += 1) {
    const loading = store.loadOlderMessages()
    respond('a', all)
    await loading
  }
  assert.deepEqual(ids(store), Array.from({ length: 120 }, (_, index) => index + 1))
})

test('请求期间切走又切回，迟到的刷新不截断新加载的历史', async () => {
  const store = createStore([conversation('a'), conversation('b')])
  const all = history(120)
  const openA = store.selectConversation('a')
  respond('a', all)
  await openA
  const firstPage = store.loadOlderMessages()
  respond('a', all)
  await firstPage
  assert.deepEqual(ids(store), Array.from({ length: 100 }, (_, index) => index + 21))

  // 会话 A 的收尾刷新在用户已经切走时发出（例如切走后旧流才收尾）
  store.setCurrentId('b')
  const refreshing = store.refreshMessages('a')
  // 用户切回 A 并继续向前翻页
  store.setCurrentId('a')
  const loading = store.loadOlderMessages()
  respondLatest('a', all) // 后发的翻页先返回
  await loading
  assert.deepEqual(ids(store), Array.from({ length: 120 }, (_, index) => index + 1))

  respond('a', all) // 迟到的刷新这才返回
  await refreshing

  assert.deepEqual(ids(store), Array.from({ length: 120 }, (_, index) => index + 1))
})

// ---------------------------------------------------------------------------
// 会话标题检索（issue #251）：检索词只影响会话列表请求，消息加载与流式收尾不受牵连。
// 桩按发出顺序记录每次 getConversations 的参数，因此能直接断言 q / 游标到底带了什么。
// ---------------------------------------------------------------------------

const CONVERSATION_FETCH = 51
const CONVERSATION_PAGE = 50

const conversationRow = (id, title = id, second = 0) => ({
  id,
  title,
  knowledge_base_id: null,
  updated_at: `2026-09-01T10:00:${String(second).padStart(2, '0')}`,
})

const conversationRows = (prefix, count) =>
  Array.from({ length: count }, (_, index) =>
    conversationRow(`${prefix}-${index}`, `${prefix} ${index}`, 59 - index)
  )

const conversationIds = (store) => store.conversations.map((item) => item.id)

test('设置检索词后首页请求带 q（去首尾空白），并从新结果重建游标', async () => {
  const store = createStore([])
  const initial = store.fetchConversations()
  respondConversations(conversationRows('base', CONVERSATION_FETCH))
  await initial
  assert.equal(store.hasMoreConversations, true)
  const cursorBeforeSearch = conversationRequests[0]
  assert.equal(cursorBeforeSearch.params.q, undefined)

  const searched = store.setConversationQuery('  知识库  ')
  // store 保留原始输入供输入框回显，发请求时才是去空白后的检索词。
  assert.equal(store.conversationQuery, '  知识库  ')
  assert.deepEqual(conversationRequests.at(-1).params, {
    limit: CONVERSATION_FETCH,
    q: '知识库',
  })
  respondConversations([conversationRow('hit-only', '知识库配置')])
  await searched

  assert.deepEqual(conversationIds(store), ['hit-only'])
  // 上一轮未过滤列表留下的游标与「还有更多」必须清掉：新结果只有一条，
  // 若沿用旧游标，下一次翻页会把不相干的默认列表接在搜索结果后面。
  assert.equal(store.hasMoreConversations, false)
  const requestsBeforeLoadMore = conversationRequests.length
  await store.loadMoreConversations()
  assert.equal(conversationRequests.length, requestsBeforeLoadMore)
})

test('搜索结果的「加载更早」携带当前检索词与新游标', async () => {
  const store = createStore([])
  const hits = conversationRows('hit', CONVERSATION_FETCH)
  const searched = store.setConversationQuery('知识库')
  respondConversations(hits)
  await searched
  assert.deepEqual(store.conversations.map((item) => item.id), hits.slice(0, CONVERSATION_PAGE).map((item) => item.id))
  assert.equal(store.hasMoreConversations, true)

  const loading = store.loadMoreConversations()

  assert.deepEqual(conversationRequests.at(-1).params, {
    limit: CONVERSATION_FETCH,
    before_updated_at: hits[CONVERSATION_PAGE - 1].updated_at,
    before_id: hits[CONVERSATION_PAGE - 1].id,
    q: '知识库',
  })
  respondConversation(conversationRequests.length - 1, [])
  await loading
})

test('快速改检索词时，先发出的旧响应不得覆盖新结果', async () => {
  const store = createStore([])

  const oldRequest = store.setConversationQuery('旧')
  const newRequest = store.setConversationQuery('新')
  assert.deepEqual(conversationRequests.map((entry) => entry.params.q), ['旧', '新'])

  // 新请求先返回，旧请求后返回：只有新结果可以落到列表上。
  respondConversation(1, [conversationRow('new-hit', '新结果')])
  await newRequest
  respondConversation(0, [conversationRow('old-hit', '旧结果')])
  await oldRequest

  assert.equal(store.conversationQuery, '新')
  assert.deepEqual(conversationIds(store), ['new-hit'])
  assert.equal(store.loading, false)
})

test('改检索词后，在途的「加载更早」不会把旧检索的结果追加进来', async () => {
  const store = createStore([])
  const initial = store.fetchConversations()
  respondConversations(conversationRows('base', CONVERSATION_FETCH))
  await initial

  const loadingMore = store.loadMoreConversations()
  assert.equal(conversationRequests.length, 2)
  const searched = store.setConversationQuery('新')
  assert.equal(conversationRequests.length, 3)

  // 先返回新检索的第一页，再让旧检索的「加载更早」迟到。
  respondConversation(2, [conversationRow('new-hit', '新结果')])
  await searched
  respondConversation(1, conversationRows('old-page', 3))
  await loadingMore

  assert.deepEqual(conversationIds(store), ['new-hit'])
  assert.equal(store.loadingMoreConversations, false)
})

test('清空检索词后恢复默认列表，请求不再带 q', async () => {
  const store = createStore([])
  const searched = store.setConversationQuery('知识库')
  respondConversations([conversationRow('hit', '知识库配置')])
  await searched
  assert.deepEqual(conversationIds(store), ['hit'])

  const cleared = store.setConversationQuery('')
  assert.deepEqual(conversationRequests.at(-1).params, { limit: CONVERSATION_FETCH })
  respondConversations([conversationRow('a', '会话一'), conversationRow('b', '会话二')])
  await cleared

  assert.equal(store.conversationQuery, '')
  assert.deepEqual(conversationIds(store), ['a', 'b'])
  assert.equal(store.hasMoreConversations, false)
})

test('检索词变化时清空历史管理模式下已选中的会话', async () => {
  const store = createStore([conversation('c1'), conversation('c2'), conversation('c3')])
  store.enterHistoryManageMode()
  store.toggleConversationSelection('c1')
  store.toggleConversationSelection('c2')
  assert.deepEqual(store.selectedConversationIds, ['c1', 'c2'])

  const searched = store.setConversationQuery('报表')
  respondConversations([conversationRow('hit', '报表统计')])
  await searched

  // 搜索结果只剩一条命中项，旧选中项已不在列表里：不清空的话「已选 2 / 1」
  // 会让批量删除删掉用户根本看不见的 c1/c2。
  assert.deepEqual(store.selectedConversationIds, [])
  assert.deepEqual(conversationIds(store), ['hit'])
})

test('检索词未实际变化时不清空选中（幂等调用不误清）', async () => {
  const store = createStore([conversation('c1')])
  const searched = store.setConversationQuery('报表')
  respondConversations([conversationRow('hit', '报表统计')])
  await searched
  store.toggleConversationSelection('hit')
  assert.deepEqual(store.selectedConversationIds, ['hit'])

  const requestsBefore = conversationRequests.length
  // 仅首尾空白不同 = 同一个检索词：短路返回，不发请求，也不能动选中集合。
  await store.setConversationQuery('  报表  ')

  assert.equal(conversationRequests.length, requestsBefore)
  assert.deepEqual(store.selectedConversationIds, ['hit'])
})

test('搜索后全选只覆盖当前可见的命中项', async () => {
  const store = createStore([conversation('c1'), conversation('c2'), conversation('c3')])
  store.enterHistoryManageMode()

  const searched = store.setConversationQuery('报表')
  respondConversations([conversationRow('hit-1', '报表一'), conversationRow('hit-2', '报表二')])
  await searched

  store.toggleSelectAllConversations()

  assert.deepEqual(store.selectedConversationIds, ['hit-1', 'hit-2'])
})

test('检索在途时全选：首页结果落地后选中集合只保留可见项', async () => {
  const store = createStore([conversation('c1'), conversation('c2')])
  store.enterHistoryManageMode()

  const searched = store.setConversationQuery('报表')
  // 请求还在途，列表仍是旧结果：此刻「全选」选中是即将被替换掉的 c1 / c2。
  assert.equal(store.loading, true)
  store.toggleSelectAllConversations()
  assert.deepEqual(store.selectedConversationIds, ['c1', 'c2'])

  respondConversations([conversationRow('hit', '报表统计')])
  await searched

  assert.deepEqual(conversationIds(store), ['hit'])
  // 响应落地、列表被整体替换：选中集合必须与可见列表对齐，
  // 否则界面会显示「已选 2 / 1」，批量删除会删掉用户看不见的 c1 / c2。
  assert.deepEqual(store.selectedConversationIds, [])
  assert.equal(store.selectedConversationIds.length, 0)
  assert.ok(
    store.selectedConversationIds.length <= store.conversations.length,
    '选中计数不得超过可见列表条数'
  )
})

test('首页刷新后选中集合只保留仍然可见的会话', async () => {
  const store = createStore([conversation('c1'), conversation('c2')])
  store.enterHistoryManageMode()
  store.toggleSelectAllConversations()
  assert.deepEqual(store.selectedConversationIds, ['c1', 'c2'])

  const refreshing = store.fetchConversations()
  respondConversations([conversationRow('c1', '仍在列表'), conversationRow('c3', '新增命中')])
  await refreshing

  // 只过滤、不清空：仍在列表里的 c1 保留选中，消失的 c2 移除。
  assert.deepEqual(conversationIds(store), ['c1', 'c3'])
  assert.deepEqual(store.selectedConversationIds, ['c1'])
})

test('迟到的旧首页响应不改动选中集合（只对齐最新响应）', async () => {
  const store = createStore([conversation('c1'), conversation('c2')])
  store.enterHistoryManageMode()
  store.toggleSelectAllConversations()

  const stale = store.fetchConversations()
  const latest = store.fetchConversations()
  respondConversation(1, [conversationRow('hit', '新结果')])
  await latest
  assert.deepEqual(store.selectedConversationIds, [])

  store.toggleConversationSelection('hit')
  respondConversation(0, [conversationRow('c1', '旧结果'), conversationRow('c2', '旧结果')])
  await stale

  // 旧响应整段丢弃：列表与选中都保持最新响应落地后的状态。
  assert.deepEqual(conversationIds(store), ['hit'])
  assert.deepEqual(store.selectedConversationIds, ['hit'])
})

