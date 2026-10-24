import request from './request'

export const knowledgeAPI = {
  getList(params) {
    return request.get('/knowledge', { params })
  },
  // params 支持 { limit, after_id }：不传时后端只返回第一页（按创建顺序，旧 -> 新）。
  getBases(params) {
    return request.get('/knowledge-bases', { params })
  },
  createBase(name, config = {}) {
    return request.post('/knowledge-bases', { name }, config)
  },
  renameBase(id, name, config = {}) {
    return request.put(`/knowledge-bases/${id}`, { name }, config)
  },
  deleteBase(id, config = {}) {
    return request.delete(`/knowledge-bases/${id}`, config)
  },
  getDetail(id, config = {}) {
    // config 透传：来源卡片跳转用 { silent: true }，不存在/失败由调用方给文案。
    return request.get(`/knowledge/${id}`, config)
  },
  getContent(id, config = {}) {
    // config 透传：详情预览用 { silent: true } 抑制拦截器的顶部提示，
    // 失败文案只由预览区给出，避免同一次失败提示两遍。
    return request.get(`/knowledge/${id}/content`, config)
  },
  upload(file, knowledgeBaseId, onProgress, config = {}) {
    const formData = new FormData()
    formData.append('file', file)
    if (knowledgeBaseId) {
      formData.append('knowledge_base_id', knowledgeBaseId)
    }

    return request.post('/knowledge/upload', formData, {
      ...config,
      headers: { ...config.headers, 'Content-Type': 'multipart/form-data' },
      onUploadProgress: onProgress,
    })
  },
  delete(id, config = {}) {
    return request.delete(`/knowledge/${id}`, config)
  },
  async batchDelete(ids, config = {}) {
    const results = await Promise.allSettled(ids.map((id) => request.delete(`/knowledge/${id}`, config)))
    const failed = results.filter((item) => item.status === 'rejected')
    return {
      total: ids.length,
      succeeded: results.length - failed.length,
      failed: failed.length,
      errors: failed.map((item) => item.reason),
    }
  },
}
