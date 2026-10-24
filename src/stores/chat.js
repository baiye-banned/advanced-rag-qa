import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import { chatAPI, streamChat } from '@/api/chat'
import { RAGAS_STATUS, isPendingRagasStatus } from '@/utils/ragasStatus'

const EVALUATION_POLL_INTERVAL_MS = 3000
const EVALUATION_POLL_TIMEOUT_MS = 190000

// 会话消息分页：与后端 CHAT_MESSAGE_DEFAULT_LIMIT / CHAT_MESSAGE_MAX_LIMIT 保持一致。
// 后端一次最多返回 MESSAGE_PAGE_SIZE 条，UI 先展示最新一页，再按需向前翻。
const MESSAGE_PAGE_SIZE = 50
// 取页时多要一条：用「是否多出来」判断还有没有更早的消息。
// 只取一页时无法区分「正好取满」和「已经取完」，会给出一个点了没反应的按钮。
const MESSAGE_FETCH_LIMIT = MESSAGE_PAGE_SIZE + 1

// 会话列表分页：与后端 LIST_DEFAULT_LIMIT / LIST_MAX_LIMIT 保持一致（issue #191）。
// 侧栏原先一次取回全部会话，接口加上限之后必须给出按需加载的入口，
// 否则超出上限的会话在界面上就是静默消失。
const CONVERSATION_PAGE_SIZE = 50
const CONVERSATION_FETCH_LIMIT = CONVERSATION_PAGE_SIZE + 1

// 把「最新 N+1 条」切成「页面」与「还有更早」：后端按旧 -> 新返回，
// 多出来的那条是最旧的一条，丢掉它，剩下正好一页。
function splitPage(rows) {
  const hasMore = rows.length > MESSAGE_PAGE_SIZE
  return {
    page: hasMore ? rows.slice(rows.length - MESSAGE_PAGE_SIZE) : rows,
    hasMore,
  }
}

// 会话列表按「最近活动」倒序返回：多出来的那条是最旧的一条（在末尾），丢掉它剩下正好一页。
function splitConversationPage(rows) {
  const hasMore = rows.length > CONVERSATION_PAGE_SIZE
  return {
    page: hasMore ? rows.slice(0, CONVERSATION_PAGE_SIZE) : rows,
    hasMore,
  }
}

// 翻页游标：后端按 (updated_at, id) 复合键翻页（会话主键是 uuid，排不出先后；
// updated_at 只到秒，同秒时按主键定序），所以两样都要回带。
function conversationCursorOf(page) {
  const last = page[page.length - 1]
  if (!last?.id || !last?.updated_at) return null
  return { id: last.id, updated_at: last.updated_at }
}

export const useChatStore = defineStore('chat', () => {
  const conversations = ref([])
  const currentId = ref(null)
  const messages = ref([])
  const loading = ref(false)
  const hasMoreConversations = ref(false)
  const loadingMoreConversations = ref(false)
  const hasMoreMessages = ref(false)
  const loadingOlderMessages = ref(false)
  const historyManageMode = ref(false)
  const selectedConversationIds = ref([])

  const streaming = ref(false)
  const streamContent = ref('')
  const streamSources = ref([])
  const streamTrace = ref({ trace_id: '', status: 'running', events: [] })
  const streamingConversationId = ref(null)
  const pendingRouteConversationId = ref(null)
  const selectedKnowledgeBaseId = ref(null)
  const errorMessage = ref('')
  const abortController = ref(null)
  const streamingHasAttachments = ref(false)
  const streamImageAnalysis = ref(null)
  const evaluationPollTimer = ref(null)

  // 请求序号：切换会话会让在途的「会话消息加载」失效；新的提问会让上一次「流式请求」的回调失效，
  // 避免陈旧响应覆盖当前会话。
  let loadSeq = 0
  let streamSeq = 0
  // 消息窗口是否被截断过（replaceMessages 砍掉后半段），见 refreshMessages 的重建分支。
  let windowTruncated = false
  // 会话列表已加载区间的末位游标（{ id, updated_at }），loadMoreConversations 用它向后翻。
  let conversationCursor = null
  // 视图世代：用户显式清空会话视图（新建对话 / 删除会话）时递增。在途流据此放弃「认领新建会话」，
  // 否则用户已经开了新对话，旧流收尾还会把他拽回原会话。
  let viewEpoch = 0

  const currentConversation = computed(() =>
    conversations.value.find((conversation) => conversation.id === currentId.value)
  )

  // 取最新一页会话。收尾（handleStreamDone）与侧栏挂载都走这里：只刷新最前面这一页，
  // 更早的会话由 loadMoreConversations 按需向后翻——整表取回正是 #191 要修的那件事。
  async function fetchConversations() {
    loading.value = true
    try {
      const response = await chatAPI.getConversations({ limit: CONVERSATION_FETCH_LIMIT })
      const rows = Array.isArray(response) ? response : []
      const { page, hasMore } = splitConversationPage(rows)
      conversations.value = page
      conversationCursor = conversationCursorOf(page)
      hasMoreConversations.value = hasMore && conversationCursor !== null
      return conversations.value
    } finally {
      loading.value = false
    }
  }

  // 向后翻页：以已加载区间的最后一条作游标，把更早的一页接在列表末尾。
  // 后端按同一游标返回，翻页不会重复也不会漏；取不满一页即说明已经到最早一条。
  async function loadMoreConversations() {
    if (!hasMoreConversations.value || loadingMoreConversations.value || !conversationCursor) return
    loadingMoreConversations.value = true
    try {
      const response = await chatAPI.getConversations({
        limit: CONVERSATION_FETCH_LIMIT,
        before_updated_at: conversationCursor.updated_at,
        before_id: conversationCursor.id,
      })
      const rows = Array.isArray(response) ? response : []
      const { page, hasMore } = splitConversationPage(rows)
      // 逐条去重：流式收尾新插入的会话、以及刷新期间被改动的会话都可能与这一页重叠。
      const known = new Set(conversations.value.map((item) => item.id))
      const fresh = page.filter((item) => item?.id && !known.has(item.id))
      if (fresh.length) {
        conversations.value = [...conversations.value, ...fresh]
      }
      const nextCursor = conversationCursorOf(page)
      if (nextCursor) conversationCursor = nextCursor
      hasMoreConversations.value = hasMore && nextCursor !== null
    } finally {
      loadingMoreConversations.value = false
    }
  }

  function upsertConversation(conversation) {
    if (!conversation?.id) return
    const index = conversations.value.findIndex((item) => item.id === conversation.id)
    const next = {
      id: conversation.id,
      title: conversation.title || '未命名对话',
      knowledge_base_id: conversation.knowledge_base_id || null,
      knowledge_base_name: conversation.knowledge_base_name || '',
      created_at: conversation.created_at || '',
      updated_at: conversation.updated_at || '',
    }
    if (index >= 0) {
      conversations.value[index] = { ...conversations.value[index], ...next }
    } else {
      conversations.value.unshift(next)
    }
  }

  // 只有「最新一次加载」且「仍停留在该会话」的响应才允许写入状态。
  function isLatestLoad(seq, id) {
    return seq === loadSeq && currentId.value === id
  }

  async function selectConversation(id) {
    const seq = ++loadSeq
    currentId.value = id
    loading.value = true
    try {
      // 只取最新一页：历史由 loadOlderMessages 按需向前翻，避免一次拉回整段会话。
      const response = await chatAPI.getMessages(id, { limit: MESSAGE_FETCH_LIMIT })
      if (!isLatestLoad(seq, id)) return
      const rows = Array.isArray(response) ? response.map(normalizeMessage) : []
      const { page, hasMore } = splitPage(rows)
      messages.value = page
      hasMoreMessages.value = hasMore
      windowTruncated = false
      const conversation = conversations.value.find((item) => item.id === id)
      if (conversation?.knowledge_base_id) {
        selectedKnowledgeBaseId.value = conversation.knowledge_base_id
      }
      if (hasPendingEvaluation(messages.value)) {
        startEvaluationPolling(id)
      } else {
        stopEvaluationPolling()
      }
    } finally {
      if (seq === loadSeq) loading.value = false
    }
  }

  // 向前翻页：以当前最旧一条的 id 作游标，把更早的一页拼到列表头部。
  // 后端按 id 游标返回，翻页不会重复也不会漏；取不满一页即说明已经到最早一条。
  async function loadOlderMessages() {
    const id = currentId.value
    const oldest = messages.value.find((message) => message.id != null)
    if (!id || !oldest || !hasMoreMessages.value || loadingOlderMessages.value) return
    loadingOlderMessages.value = true
    try {
      const response = await chatAPI.getMessages(id, {
        limit: MESSAGE_FETCH_LIMIT,
        before_id: oldest.id,
      })
      if (currentId.value !== id) return
      const rows = Array.isArray(response) ? response.map(normalizeMessage) : []
      const { page, hasMore } = splitPage(rows)
      const known = new Set(messages.value.map((message) => message.id))
      const fresh = page.filter((message) => message.id == null || !known.has(message.id))
      if (fresh.length) {
        messages.value = [...fresh, ...messages.value]
      }
      hasMoreMessages.value = hasMore
    } finally {
      loadingOlderMessages.value = false
    }
  }

  function addMessage(message) {
    messages.value.push(normalizeMessage(message))
  }

  function replaceMessages(nextMessages) {
    messages.value = nextMessages.map(normalizeMessage)
    // 视图被截断（重新生成会砍掉后半段）后，它和后续取回的最新一页之间可能断档。
    // 记下来，下一次刷新改为重建窗口，避免在中间留下取不到的空洞。
    windowTruncated = true
  }

  function setCurrentId(id) {
    currentId.value = id || null
  }

  function setSelectedKnowledgeBaseId(id) {
    selectedKnowledgeBaseId.value = id || null
  }

  function enterHistoryManageMode() {
    historyManageMode.value = true
    selectedConversationIds.value = []
  }

  function exitHistoryManageMode() {
    historyManageMode.value = false
    selectedConversationIds.value = []
  }

  function toggleConversationSelection(id) {
    if (!id) return
    const exists = selectedConversationIds.value.includes(id)
    selectedConversationIds.value = exists
      ? selectedConversationIds.value.filter((item) => item !== id)
      : [...selectedConversationIds.value, id]
  }

  function toggleSelectAllConversations() {
    if (selectedConversationIds.value.length === conversations.value.length) {
      selectedConversationIds.value = []
      return
    }
    selectedConversationIds.value = conversations.value.map((conversation) => conversation.id)
  }

  function isConversationSelected(id) {
    return selectedConversationIds.value.includes(id)
  }

  function sendMessage(question, attachments = []) {
    const rawText = question.trim()
    const displayText = rawText || (attachments.length ? '请分析这张图片' : '')
    if ((!rawText && !attachments.length) || streaming.value) return

    errorMessage.value = ''
    streaming.value = true
    streamContent.value = ''
    streamSources.value = []
    streamTrace.value = { trace_id: '', status: 'running', events: [] }
    streamImageAnalysis.value = null
    streamingConversationId.value = currentId.value
    pendingRouteConversationId.value = null
    streamingHasAttachments.value = attachments.length > 0
    abortController.value = new AbortController()
    const requestSeq = ++streamSeq
    const viewEpochAtSend = viewEpoch

    addMessage({ role: 'user', content: displayText, attachments })

    streamChat({
      conversationId: currentId.value,
      knowledgeBaseId: selectedKnowledgeBaseId.value,
      question: rawText,
      attachments,
      signal: abortController.value.signal,
      onMessage: (content, event) => {
        if (!isLatestStream(requestSeq)) return
        if (event?.type === 'conversation') {
          const conversation = event.conversation || {}
          // 归属必须在改动状态之前判定：流式过程中用户可能已经切到别的会话，
          // 此时迟到的 conversation 事件只能更新会话列表，不得改写当前会话的知识库选择。
          const belongsToCurrent =
            viewEpochAtSend === viewEpoch &&
            (!currentId.value || currentId.value === streamingConversationId.value)
          upsertConversation(conversation)
          streamingConversationId.value = conversation.id || streamingConversationId.value
          // 与下面的「认领会话」同理：视图已被用户显式清空时，也不得登记待跳转会话，
          // 否则 Chat.vue 的 watcher 会 replace 回旧会话，把用户从「新对话」拽回去。
          if (viewEpochAtSend === viewEpoch) {
            pendingRouteConversationId.value = conversation.id || null
          }
          if (!currentId.value && conversation.id && viewEpochAtSend === viewEpoch) {
            currentId.value = conversation.id
          }
          if (conversation.knowledge_base_id && belongsToCurrent) {
            selectedKnowledgeBaseId.value = conversation.knowledge_base_id
          }
          return
        }
        if (event?.type === 'sources') {
          streamSources.value = event.sources || []
          return
        }
        if (event?.type === 'trace') {
          mergeStreamTrace(event)
          return
        }
        if (event?.type === 'image_analysis') {
          streamImageAnalysis.value = normalizeImageAnalysis(event.analysis || event)
          return
        }
        if (event?.type === 'reset') {
          // 首个模型中途失败、后备模型从头重新生成：作废已渲染的增量。
          // 只清空正文缓冲，流式占位（streaming / streamSources / streamTrace）保持不变，
          // 后备模型后续的增量会照常追加到这个占位上。
          streamContent.value = ''
          return
        }
        streamContent.value += content
      },
      onDone: () => handleStreamDone(requestSeq, viewEpochAtSend),
      onError: (error) => handleStreamError(error, requestSeq),
    })
  }

  // 只有最新一次流式请求的回调才允许写状态；旧的流既不能追加内容，也不能收尾。
  function isLatestStream(seq) {
    return seq === streamSeq
  }

  async function handleStreamDone(requestSeq, viewEpochAtSend) {
    if (!isLatestStream(requestSeq)) return
    const targetConversationId = streamingConversationId.value
    const shouldTrackEvaluation = shouldTrackEvaluationFromTrace()
    const shouldShowLocalMessage = streamContent.value && currentId.value === targetConversationId
    if (shouldShowLocalMessage) {
      addMessage({
        role: 'assistant',
        content: streamContent.value,
        sources: streamSources.value,
        learning_trace: cloneTrace(streamTrace.value),
        ...streamImageAnalysisFields(),
        ragas_status: shouldTrackEvaluation ? RAGAS_STATUS.PENDING : '',
        isLocal: true,
      })
    }

    finishStreaming()
    const nextConversations = await fetchConversations().catch(() => [])
    // 改写「当前会话」属于归属写入：收尾期间可能已经又发起了新的提问、或用户已经开了新对话，
    // 只有仍是当前请求、且视图没有被显式清空过，才允许改写。
    if (!currentId.value && isLatestStream(requestSeq) && viewEpochAtSend === viewEpoch) {
      if (nextConversations?.[0]?.id) {
        setCurrentId(nextConversations[0].id)
      } else if (targetConversationId) {
        setCurrentId(targetConversationId)
      }
    }
    // 消息刷新与评测轮询按「会话归属」判定：只要用户仍停留在该会话就要刷新。若这里改按请求序号判定，
    // 「回答刚结束又立刻追问」时收尾会被新请求作废，刚生成的回答会一直停在「评测中」且没有轮询。
    if (targetConversationId && currentId.value === targetConversationId) {
      // 收尾期间用户可能又提了新问题：刷新照做（轮询要能启动），但写入快照属于请求归属写入，
      // 被新请求取代时只返回合并结果、不改动消息列表，避免抹掉刚发出的提问。
      const nextMessages = await refreshMessages(targetConversationId, {
        shouldWrite: () => isLatestStream(requestSeq),
      }).catch(() => messages.value)
      // 刷新期间用户可能已经切走，回来后不得再改动（启停评测轮询）不属于自己的会话。
      if (currentId.value !== targetConversationId) return
      if (hasPendingEvaluation(nextMessages)) {
        startEvaluationPolling(targetConversationId)
      } else {
        stopEvaluationPolling()
      }
    }
  }

  function handleStreamError(error, requestSeq) {
    if (!isLatestStream(requestSeq)) return
    if (error?.name === 'AbortError') {
      if (streamContent.value && currentId.value === streamingConversationId.value) {
        addMessage({
          role: 'assistant',
          content: `${streamContent.value}\n\n*(已停止生成)*`,
          sources: streamSources.value,
          learning_trace: cloneTrace(streamTrace.value),
          isLocal: true,
        })
      }
    } else {
      const message = getStreamErrorMessage(error)
      errorMessage.value = message
      if (currentId.value === streamingConversationId.value) {
        addMessage({
          role: 'assistant',
          content: streamContent.value || message,
          sources: streamSources.value,
          stream_error: message,
          learning_trace: cloneTrace(streamTrace.value),
          ...streamImageAnalysisFields(),
          isLocal: true,
        })
      }
    }
    finishStreaming()
  }

  function stopGeneration() {
    abortController.value?.abort()
  }

  // 中止在途流并作废它的回调（清空视图 / 删除会话时使用）。
  // 先递增 streamSeq：旧流的 onMessage / onDone / onError 立刻失去写入资格，
  // 包括 abort 触发的 AbortError 收尾，不会往已清空的视图补「已停止生成」。
  // 再 abort 真正的传输，最后复位流状态，让 streaming 立即变为 false——
  // 否则新对话的第一条消息会被 sendMessage 的 streaming 守卫挡住。
  function abortAndDiscardStream() {
    streamSeq += 1
    const controller = abortController.value
    controller?.abort()
    finishStreaming()
  }

  function finishStreaming() {
    streaming.value = false
    streamContent.value = ''
    streamSources.value = []
    streamTrace.value = { trace_id: '', status: 'running', events: [] }
    streamImageAnalysis.value = null
    streamingConversationId.value = null
    pendingRouteConversationId.value = null
    streamingHasAttachments.value = false
    abortController.value = null
  }

  function getStreamErrorMessage(error) {
    if (error?.message) return error.message
    if (streamingHasAttachments.value) {
      return '图片内容处理失败，请检查视觉模型配置或图片地址是否可访问'
    }
    return '回答生成失败，请稍后重试'
  }

  function normalizeImageAnalysis(analysis = {}) {
    return {
      status: analysis.status || analysis.image_analysis_status || '',
      description: analysis.description || analysis.image_description || '',
      error: analysis.error || analysis.image_analysis_error || '',
    }
  }

  function streamImageAnalysisFields() {
    const analysis = normalizeImageAnalysis(streamImageAnalysis.value || {})
    return {
      image_analysis_status: analysis.status,
      image_analysis_error: analysis.error,
      image_description: analysis.description,
      retrieval_trace: {
        image_analysis_status: analysis.status,
        image_analysis_error: analysis.error,
        image_description: analysis.description,
      },
    }
  }

  function mergeStreamTrace(event = {}) {
    const traceId = event.trace_id || streamTrace.value.trace_id || ''
    const nextEvents = [...(streamTrace.value.events || [])]
    if (event.event) {
      const exists = nextEvents.some((item) => item.index === event.event.index)
      if (!exists) {
        nextEvents.push(event.event)
      }
    }
    streamTrace.value = {
      trace_id: traceId,
      status: event.status || streamTrace.value.status || 'running',
      events: nextEvents.sort((a, b) => Number(a.index || 0) - Number(b.index || 0)),
    }
  }

  function cloneTrace(trace = {}) {
    return {
      trace_id: trace.trace_id || '',
      status: trace.status || '',
      events: Array.isArray(trace.events) ? trace.events.map((event) => ({ ...event })) : [],
    }
  }

  function shouldTrackEvaluationFromTrace(trace = streamTrace.value) {
    const events = Array.isArray(trace?.events) ? trace.events : []
    return events.some((event) => event?.stage === 'retrieval_completed') ||
      events.some((event) => event?.stage === 'rag_gate_decided' && event?.result?.need_rag !== false)
  }

  async function deleteConversation(id) {
    await chatAPI.deleteConversation(id)
    conversations.value = conversations.value.filter((conversation) => conversation.id !== id)
    selectedConversationIds.value = selectedConversationIds.value.filter((item) => item !== id)
    if (currentId.value === id) {
      // clearMessages 会中止并丢弃在途流，无需再单独 stopGeneration。
      clearMessages()
    } else if (streamingConversationId.value === id) {
      // 删除的不是当前视图里的会话（流式期间用户切走了）：同样中止并丢弃，
      // 与手动「停止生成」不同，这里没有可保留的本地回答。
      abortAndDiscardStream()
    }
  }

  // options.shouldWrite：可选的额外写入条件，用于「收尾刷新」这类需要按请求归属判定的调用方；
  // 默认只按会话归属判定（用户还在该会话就写入），保持既有调用行为不变。
  async function refreshMessages(id = currentId.value, options = {}) {
    const { shouldWrite = null } = options
    if (!id) return []
    // 刷新只取最新一页：已翻出的更早历史按 id 拼回列表头部，否则轮询刷新会把用户
    // 翻过的历史截断。取页大小固定为一页，避免每次轮询都把已加载的整段历史搬回来。
    const response = await chatAPI.getMessages(id, { limit: MESSAGE_FETCH_LIMIT })
    const rows = (Array.isArray(response) ? response : []).map(normalizeMessage)
    const { page } = splitPage(rows)
    // isCurrent 要在 await 之后再读一次：请求期间用户可能切换会话又切回来，
    // 那时窗口里已经是新加载的内容，按请求发起时的判断合并会把它们截掉。
    const isCurrent = id === currentId.value
    const oldestPageId = page.length ? page[0].id : null
    const loadedOlder = isCurrent && !windowTruncated && oldestPageId != null
      ? messages.value.filter((message) => message.id != null && message.id < oldestPageId)
      : []
    const localAssistantMessages = messages.value.filter(
      (message) => message.role === 'assistant' && message.isLocal
    )
    const mergedMessages = [...loadedOlder, ...page]
    const latestBackendAssistant = [...mergedMessages].reverse().find(
      (message) => message.role === 'assistant'
    )
    for (const localMessage of localAssistantMessages) {
      const exists = mergedMessages.some(
        (item) =>
          item.role === localMessage.role &&
          (
            item.content === localMessage.content ||
            localMessage.content.includes(item.content) ||
            item.content.includes(localMessage.content)
          )
      )
      if (!exists && latestBackendAssistant && localMessage.ragas_status === RAGAS_STATUS.PENDING) {
        continue
      }
      if (!exists) {
        mergedMessages.push(localMessage)
      }
    }
    if (currentId.value === id && (!shouldWrite || shouldWrite())) {
      messages.value = mergedMessages
      if (isCurrent && windowTruncated) {
        // 窗口刚被重建（视图之前被截断过），从这一页重新开始向前翻。
        hasMoreMessages.value = rows.length > MESSAGE_PAGE_SIZE
      }
      // 其余情况不动 hasMoreMessages：刷新只取固定的一页，判断不出「还有没有更早的」，
      // 这个标记交给 selectConversation / loadOlderMessages 维护。
      windowTruncated = false
    }
    return mergedMessages
  }

  function hasPendingEvaluation(nextMessages = messages.value) {
    return nextMessages.some((message) =>
      message.role === 'assistant' && isPendingEvaluationStatus(message.ragas_status)
    )
  }

  function startEvaluationPolling(conversationId) {
    stopEvaluationPolling()
    let elapsed = 0
    evaluationPollTimer.value = window.setInterval(async () => {
      elapsed += EVALUATION_POLL_INTERVAL_MS
      if (currentId.value !== conversationId) {
        stopEvaluationPolling()
        return
      }
      const nextMessages = await refreshMessages(conversationId).catch(() => messages.value)
      if (elapsed > EVALUATION_POLL_TIMEOUT_MS) {
        markLocalEvaluationTimeout()
        stopEvaluationPolling()
        return
      }
      if (!hasPendingEvaluation(nextMessages)) {
        stopEvaluationPolling()
      }
    }, EVALUATION_POLL_INTERVAL_MS)
  }

  function stopEvaluationPolling() {
    if (evaluationPollTimer.value) {
      window.clearInterval(evaluationPollTimer.value)
      evaluationPollTimer.value = null
    }
  }

  function markLocalEvaluationTimeout() {
    messages.value = messages.value.map((message) => {
      if (
        message.role === 'assistant' &&
        message.isLocal &&
        isPendingEvaluationStatus(message.ragas_status)
      ) {
        return {
          ...message,
          ragas_status: RAGAS_STATUS.FAILED,
          ragas_error: '评测未及时返回，稍后刷新会话可查看最终状态',
        }
      }
      return message
    })
  }

  async function renameConversation(id, title) {
    await chatAPI.renameConversation(id, title)
    const conversation = conversations.value.find((item) => item.id === id)
    if (conversation) {
      conversation.title = title
    }
  }

  async function setMessageFeedback(messageId, feedback) {
    const previous = messages.value.find((m) => m.id === messageId)?.feedback ?? 0
    // 乐观更新：先落态，请求失败再回滚（回滚用捕获到的 previous，不用"取反"，
    // 避免快速连点下把中间态当成原值）。
    messages.value = messages.value.map((m) =>
      m.id === messageId ? { ...m, feedback } : m
    )
    try {
      await chatAPI.setMessageFeedback(messageId, feedback)
    } catch (error) {
      messages.value = messages.value.map((m) =>
        m.id === messageId ? { ...m, feedback: previous } : m
      )
      throw error
    }
  }

  function clearMessages() {
    // 先中止在途流：streaming 若不复位，新对话的第一条消息发不出去；
    // 同时作废旧流回调，避免它把内容或「已停止生成」补进已清空的视图。
    abortAndDiscardStream()
    // 用户显式离开当前视图：在途流的「认领会话」「改写知识库」写入随之失效。
    viewEpoch += 1
    currentId.value = null
    messages.value = []
    hasMoreMessages.value = false
    stopEvaluationPolling()
  }

  function normalizeMessage(message) {
    const retrievalTrace = message.retrieval_trace || {}
    const learningTrace = message.learning_trace || retrievalTrace.learning_trace || {}
    return {
      id: message.id || null,
      role: message.role,
      content: message.content,
      sources: message.sources || [],
      attachments: message.attachments || [],
      ragas_status: message.ragas_status || '',
      ragas_scores: message.ragas_scores || {},
      ragas_error: message.ragas_error || '',
      stream_error: message.stream_error || '',
      feedback: message.feedback || 0,
      retrieval_trace: retrievalTrace,
      learning_trace: learningTrace,
      rag_gate: retrievalTrace.rag_gate || {},
      route_mode: retrievalTrace.mode || '',
      trace_id: message.trace_id || learningTrace.trace_id || '',
      image_analysis_status: message.image_analysis_status || retrievalTrace.image_analysis_status || '',
      image_analysis_error: message.image_analysis_error || retrievalTrace.image_analysis_error || '',
      image_description: message.image_description || retrievalTrace.image_description || '',
      isLocal: Boolean(message.isLocal),
      created_at: message.created_at || '',
    }
  }

  function isPendingEvaluationStatus(status) {
    return isPendingRagasStatus(status)
  }

  return {
    conversations,
    currentId,
    messages,
    loading,
    hasMoreConversations,
    loadingMoreConversations,
    hasMoreMessages,
    loadingOlderMessages,
    historyManageMode,
    selectedConversationIds,
    streaming,
    streamContent,
    streamSources,
    streamTrace,
    streamingConversationId,
    pendingRouteConversationId,
    selectedKnowledgeBaseId,
    errorMessage,
    currentConversation,
    fetchConversations,
    loadMoreConversations,
    selectConversation,
    loadOlderMessages,
    addMessage,
    replaceMessages,
    refreshMessages,
    setCurrentId,
    setSelectedKnowledgeBaseId,
    enterHistoryManageMode,
    exitHistoryManageMode,
    toggleConversationSelection,
    toggleSelectAllConversations,
    isConversationSelected,
    sendMessage,
    stopGeneration,
    finishStreaming,
    deleteConversation,
    renameConversation,
    setMessageFeedback,
    clearMessages,
  }
})
