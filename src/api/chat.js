import request from './request'
import { assertStreamResponse } from '@/utils/streamResponse'
import { readStreamEvents } from '@/utils/streamEvents'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api'

export const chatAPI = {
  // params 支持 { limit, before_updated_at, before_id }：不传时后端只返回最新一页，
  // before_* 是 (updated_at, id) 复合游标，两个要么都给、要么都不给。
  getConversations(params) {
    return request.get('/chat/conversations', { params })
  },
  // params 支持 { limit, before_id }：不传时后端只返回最新一页
  getMessages(conversationId, params) {
    return request.get(`/chat/conversations/${conversationId}`, { params })
  },
  exportConversation(conversationId) {
    return request.get(`/chat/conversations/${conversationId}/export`, {
      responseType: 'blob',
      returnFullResponse: true,
      silent: true,
    })
  },
  deleteConversation(id) {
    return request.delete(`/chat/conversations/${id}`)
  },
  // 删除单条消息（issue #256）：用户消息截断其后，助手消息只删一条。
  deleteMessage(id) {
    return request.delete(`/chat/messages/${id}`)
  },
  renameConversation(id, title) {
    return request.put(`/chat/conversations/${id}`, { title })
  },
  getTrace(traceId) {
    return request.get(`/chat/traces/${traceId}`)
  },
  getMessageTrace(messageId) {
    return request.get(`/chat/messages/${messageId}/trace`)
  },
  // 消息级反馈（issue #250）：1=赞 / -1=踩 / 0=取消。
  setMessageFeedback(messageId, feedback) {
    return request.post(`/chat/messages/${messageId}/feedback`, { feedback })
  },
  uploadAttachment(file, onProgress) {
    const formData = new FormData()
    formData.append('file', file)

    return request.post('/chat/attachments', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress,
    })
  },
}

export { assertStreamResponse }

export async function streamChat({
  conversationId,
  knowledgeBaseId,
  question,
  attachments = [],
  signal,
  onMessage,
  onDone,
  onError,
}) {
  const token = localStorage.getItem('token')

  try {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify({
        conversation_id: conversationId,
        knowledge_base_id: knowledgeBaseId,
        question,
        attachments,
      }),
      signal,
    })

    await assertStreamResponse(response)

    const doneReceived = await readStreamEvents(response.body, onMessage)

    if (doneReceived) {
      onDone?.()
    } else {
      throw new Error('流式响应未正常结束')
    }
  } catch (error) {
    onError?.(error)
  }
}
