<template>
  <div class="flex h-full flex-col bg-slate-50">
    <header class="flex min-h-14 items-center justify-between gap-4 border-b border-slate-200 bg-white px-5 py-3">
      <div class="min-w-0">
        <div v-if="chatStore.currentId" class="flex items-center gap-2">
          <el-input
            v-if="renaming"
            ref="renameRef"
            v-model.trim="renameTitle"
            size="small"
            class="w-72"
            maxlength="40"
            @keyup.enter="confirmRename"
            @blur="confirmRename"
          />
          <h2 v-else class="truncate text-sm font-medium text-slate-800">
            {{ chatStore.currentConversation?.title || '未命名对话' }}
          </h2>
          <el-tooltip v-if="!renaming" content="重命名" placement="bottom">
            <el-button link :icon="Edit" @click="startRename" />
          </el-tooltip>
        </div>
        <h2 v-else class="text-sm font-medium text-slate-800">新对话</h2>
      </div>

      <div class="flex flex-wrap items-center justify-end gap-3">
        <div class="rounded-xl border border-slate-200 bg-slate-50 px-4 py-2">
          <p class="text-[11px] font-medium uppercase tracking-[0.08em] text-slate-400">当前知识库</p>
          <div class="mt-1 flex flex-wrap items-center gap-3">
            <span class="max-w-[220px] truncate text-sm font-medium text-slate-700">
              {{ currentKnowledgeBaseName }}
            </span>
            <el-select
              v-model="selectedKnowledgeBaseId"
              class="w-52"
              size="small"
              :disabled="Boolean(chatStore.currentId) || chatStore.streaming"
              placeholder="选择知识库"
              @change="changeKnowledgeBase"
            >
              <el-option
                v-for="base in knowledgeBases"
                :key="base.id"
                :label="base.name"
                :value="base.id"
              />
            </el-select>
          </div>
        </div>

        <el-button
          :icon="Download"
          :loading="exporting"
          :disabled="!chatStore.currentId || chatStore.streaming || exporting"
          data-testid="export-markdown"
          @click="exportConversation"
        >
          导出 Markdown
        </el-button>
        <el-button :icon="Plus" @click="createNewConversation">新建对话</el-button>
      </div>
    </header>

    <main
      ref="chatRef"
      v-loading="isConversationLoading"
      element-loading-text="正在加载对话..."
      :aria-busy="isConversationLoading ? 'true' : 'false'"
      class="min-h-0 flex-1 overflow-y-auto px-6 py-5"
    >
      <section
        v-if="!chatStore.messages.length && !isCurrentConversationStreaming && !isConversationLoading"
        class="flex h-full items-center justify-center"
      >
        <div class="w-full max-w-2xl text-center">
          <div class="mx-auto flex h-14 w-14 items-center justify-center rounded-lg bg-brand-50 text-brand-600">
            <el-icon :size="30"><ChatDotSquare /></el-icon>
          </div>
          <h1 class="mt-5 text-2xl font-semibold text-slate-900">今天想了解什么？</h1>
          <p class="mt-2 text-sm text-slate-500">向企业知识库提问，系统会结合已上传资料生成答案。</p>
          <div class="mt-6 grid grid-cols-1 gap-3 sm:grid-cols-2">
            <button
              v-for="suggestion in CHAT_SUGGESTIONS"
              :key="suggestion"
              type="button"
              class="rounded-lg border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-700 transition-colors hover:border-brand-500 hover:text-brand-700"
              @click="askSuggestion(suggestion)"
            >
              {{ suggestion }}
            </button>
          </div>
        </div>
      </section>

      <!-- 加载在途时不渲染消息列表：旧会话内容不能被当成新会话内容展示。 -->
      <section v-else-if="!isConversationLoading" class="mx-auto max-w-5xl space-y-5">
        <div v-if="chatStore.hasMoreMessages" class="flex justify-center">
          <el-button
            text
            size="small"
            :loading="chatStore.loadingOlderMessages"
            @click="loadOlderMessages"
          >
            加载更早的消息
          </el-button>
        </div>
        <article
          v-for="(message, index) in chatStore.messages"
          :key="`${message.role}-${index}-${message.created_at || index}`"
          :class="['flex', message.role === 'user' ? 'justify-end' : 'justify-start']"
        >
          <div
            :class="[
              'group relative max-w-[82%] rounded-lg px-4 py-3 shadow-xs',
              message.role === 'user'
                ? 'bg-brand-600 text-white'
                : 'border border-slate-200 bg-white text-slate-700',
            ]"
          >
            <div v-if="message.role === 'assistant'" class="mb-2 flex items-center gap-2 text-xs text-slate-500">
              <el-icon><MagicStick /></el-icon>
              <span>AI 助手</span>
            </div>
            <template v-if="message.role === 'assistant'">
              <MarkdownRenderer :content="message.content" />
              <p
                v-if="message.stream_error"
                class="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-700"
              >
                {{ message.stream_error }}
              </p>
              <p
                v-if="shouldShowImageAnalysisWarning(message)"
                class="mt-3 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-700"
              >
                {{ imageAnalysisWarningText(message) }}
              </p>
              <button
                v-if="message.sources?.length"
                type="button"
                class="mt-3 inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-brand-500 hover:text-brand-700"
                @click="openSources(message.sources)"
              >
                <el-icon><Document /></el-icon>
                参考 {{ message.sources.length }} 篇资料
              </button>
              <button
                v-if="hasTrace(message)"
                type="button"
                class="ml-2 mt-3 inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-brand-500 hover:text-brand-700"
                @click="openTrace(message)"
              >
                <el-icon><Document /></el-icon>
                流程
              </button>
              <div
                v-if="message.ragas_status"
                class="mt-3 rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"
              >
                <div class="mb-1 flex items-center justify-between">
                  <span class="font-medium text-slate-700">RAG Evaluation</span>
                  <span :class="ragasStatusClass(message.ragas_status)">
                    {{ ragasStatusText(message.ragas_status) }}
                  </span>
                </div>
                <div v-if="message.ragas_status === RAGAS_STATUS.DONE" class="grid grid-cols-3 gap-2">
                  <div v-for="metric in RAGAS_METRICS" :key="metric.key">
                    <p class="text-slate-400">{{ metric.label }}</p>
                    <p class="mt-0.5 font-medium text-slate-800">{{ formatScore(message.ragas_scores?.[metric.key]) }}</p>
                  </div>
                </div>
                <p v-else-if="message.ragas_status === RAGAS_STATUS.FAILED" class="leading-5 text-red-500">
                  {{ message.ragas_error || '评测失败' }}
                </p>
                <p v-else class="text-slate-400">评测中...</p>
              </div>
            </template>
            <div v-else class="space-y-2">
              <div v-if="hasDisplayValue(message.content)" class="whitespace-pre-wrap text-sm leading-6">{{ message.content }}</div>
              <div v-if="message.attachments?.length" class="grid max-w-sm grid-cols-2 gap-2">
                <img
                  v-for="attachment in message.attachments"
                  :key="attachment.object_key || attachment.url"
                  :src="attachment.url"
                  :alt="attachment.name || '上传图片'"
                  class="h-28 w-full rounded-md border border-white/30 object-cover"
                />
              </div>
            </div>

            <!-- 用户消息没有悬浮操作栏，删除入口放在气泡内正文下方，避免遮住内容。 -->
            <div v-if="message.role === 'user'" class="mt-1 flex justify-end">
              <button
                type="button"
                class="inline-flex items-center gap-1 rounded px-1 py-0.5 text-xs text-white/70 transition-colors hover:text-white disabled:cursor-not-allowed disabled:opacity-40"
                data-testid="delete-message"
                data-message-role="user"
                :disabled="!message.id || chatStore.streaming"
                @click="onDeleteMessage(message)"
              >
                <el-icon><Delete /></el-icon>
                删除
              </button>
            </div>

            <div
              v-if="message.role === 'assistant'"
              class="pointer-events-none absolute right-2 top-2 flex rounded-md border border-slate-200 bg-white/95 opacity-0 shadow-xs transition-opacity group-hover:pointer-events-auto group-hover:opacity-100 focus-within:pointer-events-auto focus-within:opacity-100"
            >
              <el-tooltip content="复制" placement="bottom">
                <el-button link :icon="CopyDocument" @click="copyMessage(message.content)" />
              </el-tooltip>
              <el-tooltip content="重新生成" placement="bottom">
                <el-button link :icon="Refresh" @click="regenerate(index)" />
              </el-tooltip>
              <el-tooltip content="有帮助" placement="bottom">
                <el-button
                  link
                  :icon="Top"
                  data-testid="feedback-up"
                  :class="message.feedback === 1 ? 'text-brand-600' : 'text-slate-400'"
                  :disabled="!message.id"
                  @click="onFeedback(message, 1)"
                />
              </el-tooltip>
              <el-tooltip content="没帮助" placement="bottom">
                <el-button
                  link
                  :icon="Bottom"
                  data-testid="feedback-down"
                  :class="message.feedback === -1 ? 'text-brand-600' : 'text-slate-400'"
                  :disabled="!message.id"
                  @click="onFeedback(message, -1)"
                />
              </el-tooltip>
              <el-tooltip content="删除" placement="bottom">
                <el-button
                  link
                  :icon="Delete"
                  data-testid="delete-message"
                  data-message-role="assistant"
                  :disabled="!message.id || chatStore.streaming"
                  @click="onDeleteMessage(message)"
                />
              </el-tooltip>
            </div>
          </div>
        </article>

        <article v-if="isCurrentConversationStreaming" class="flex justify-start">
          <div class="max-w-[82%] rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-xs">
            <div class="mb-2 flex items-center gap-2 text-xs text-slate-500">
              <el-icon><Loading /></el-icon>
              <span>AI 正在生成</span>
            </div>
            <MarkdownRenderer :content="chatStore.streamContent || '正在分析问题并生成答案...'" />
            <button
              v-if="chatStore.streamSources.length"
              type="button"
              class="mt-3 inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-brand-500 hover:text-brand-700"
              @click="openSources(chatStore.streamSources)"
            >
              <el-icon><Document /></el-icon>
              参考 {{ chatStore.streamSources.length }} 篇资料
            </button>
            <button
              v-if="chatStore.streamTrace.events?.length"
              type="button"
              class="ml-2 mt-3 inline-flex items-center gap-1 rounded-md border border-slate-200 px-2 py-1 text-xs text-slate-500 hover:border-brand-500 hover:text-brand-700"
              @click="openTrace({ learning_trace: chatStore.streamTrace })"
            >
              <el-icon><Document /></el-icon>
              流程
            </button>
          </div>
        </article>

        <el-alert
          v-if="chatStore.errorMessage"
          :title="chatStore.errorMessage"
          type="error"
          show-icon
          :closable="true"
          @close="chatStore.errorMessage = ''"
        />
      </section>
    </main>

    <footer class="border-t border-slate-200 bg-white p-4">
      <div class="mx-auto max-w-5xl">
        <el-input
          v-model="question"
          type="textarea"
          :rows="3"
          resize="none"
          :disabled="chatStore.streaming"
          :placeholder="chatStore.streaming ? 'AI 正在回答...' : '输入问题，Enter 发送，Shift + Enter 换行'"
          @keydown.enter="handleInputEnter"
        />
        <div v-if="attachments.length" class="mt-3 flex flex-wrap gap-2">
          <div
            v-for="attachment in attachments"
            :key="attachment.object_key"
            class="group relative h-16 w-16 overflow-hidden rounded-md border border-slate-200 bg-slate-50"
          >
            <img :src="attachment.url" :alt="attachment.name" class="h-full w-full object-cover" />
            <button
              type="button"
              class="absolute right-1 top-1 hidden rounded bg-slate-900/70 p-0.5 text-white group-hover:block"
              @click="removeAttachment(attachment)"
            >
              <el-icon><CloseBold /></el-icon>
            </button>
          </div>
        </div>
        <div class="mt-3 flex items-center justify-between gap-3">
          <span class="text-xs text-slate-400">答案由知识库检索增强生成，请结合业务资料复核。</span>
          <div class="flex items-center gap-2">
            <el-upload
              :show-file-list="false"
              :before-upload="handleImageUpload"
              :accept="ACCEPTED_IMAGE_INPUT"
              :disabled="chatStore.streaming || uploadingAttachment"
            >
              <el-button :icon="Picture" :loading="uploadingAttachment" :disabled="chatStore.streaming">
                图片
              </el-button>
            </el-upload>
            <el-button v-if="chatStore.streaming" type="warning" :icon="CloseBold" @click="stopGeneration">
              停止生成
            </el-button>
            <el-button
              type="primary"
              :disabled="chatStore.streaming || (!question.trim() && !attachments.length)"
              @click="sendMessage"
            >
              发送
            </el-button>
          </div>
        </div>
      </div>
    </footer>

    <el-drawer v-model="sourcesVisible" title="参考资料" size="360px" append-to-body>
      <div v-if="activeSources.length" class="space-y-4">
        <article
          v-for="source in activeSources"
          :key="`${source.file_id}-${source.index}`"
          class="rounded-lg border border-slate-200 bg-white p-3"
          :class="
            sourceFileId(source)
              ? 'cursor-pointer transition-colors hover:border-brand-400 hover:bg-brand-50/40 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-brand-500'
              : ''
          "
          :role="sourceFileId(source) ? 'button' : undefined"
          :tabindex="sourceFileId(source) ? 0 : undefined"
          :aria-label="sourceFileId(source) ? `查看资料详情：${source.file_name || '未命名资料'}` : undefined"
          @click="openSourceFile(source)"
          @keydown.enter.prevent="openSourceFile(source)"
          @keydown.space.prevent="openSourceFile(source)"
        >
          <div class="flex items-start gap-2">
            <span class="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-brand-50 text-xs font-medium text-brand-700">
              {{ source.index }}
            </span>
            <div class="min-w-0">
              <h3 class="truncate text-sm font-medium text-slate-900">{{ source.file_name || '未命名资料' }}</h3>
              <p class="mt-1 whitespace-pre-wrap text-xs leading-5 text-slate-500">{{ sourceText(source) }}</p>
              <div class="mt-2 flex flex-wrap gap-1 text-[11px] text-slate-400">
                <span
                  v-if="source.rerank_score !== null && source.rerank_score !== undefined"
                  class="rounded bg-slate-100 px-1.5 py-0.5"
                >
                  rerank {{ formatScore(source.rerank_score) }}
                </span>
                <span v-if="hasDisplayValue(source.rrf_score)" class="rounded bg-slate-100 px-1.5 py-0.5">
                  RRF {{ Number(source.rrf_score).toFixed(3) }}
                </span>
                <span
                  v-for="sourceRoute in source.routes || []"
                  :key="`${source.index}-${sourceRoute.route}-${sourceRoute.rank}`"
                  class="rounded bg-slate-100 px-1.5 py-0.5"
                >
                  {{ sourceRoute.route }} #{{ sourceRoute.rank }}
                </span>
              </div>
              <p v-if="source.rerank_reason" class="mt-1 text-[11px] leading-4 text-slate-400">
                {{ source.rerank_reason }}
              </p>
            </div>
          </div>
        </article>
      </div>
      <el-empty v-else description="暂无参考资料" />
    </el-drawer>

    <el-drawer v-model="traceVisible" title="本次回答执行流程" size="min(1100px, 96vw)" append-to-body>
      <div v-loading="traceLoading" class="space-y-4">
        <div class="rounded-md border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600">
          Trace ID：{{ activeTrace.trace_id || '--' }} · 状态：{{ activeTrace.status || '--' }}
        </div>
        <el-tabs v-model="traceTab">
          <el-tab-pane label="时间线" name="timeline">
            <el-timeline v-if="traceEvents.length">
              <el-timeline-item
                v-for="event in traceEvents"
                :key="event.index"
                :timestamp="event.time"
                placement="top"
              >
                <div class="rounded-md border border-slate-200 bg-white p-3">
                  <div class="flex items-center justify-between gap-3">
                    <p class="text-sm font-medium text-slate-800">{{ event.index }}. {{ event.stage }}</p>
                    <span class="text-xs text-slate-400">{{ event.function }}</span>
                  </div>
                  <p class="mt-1 text-xs leading-5 text-slate-500">{{ event.note || '暂无说明' }}</p>
                </div>
              </el-timeline-item>
            </el-timeline>
            <el-empty v-else description="暂无流程事件" />
          </el-tab-pane>

          <el-tab-pane label="变量流" name="variables">
            <TraceVariableFlow :trace="activeTrace" />
          </el-tab-pane>

          <el-tab-pane label="检索过程" name="retrieval">
            <el-collapse v-if="retrievalRoutes.length">
              <el-collapse-item
                v-for="retrievalRoute in retrievalRoutes"
                :key="retrievalRoute.route"
                :title="`${retrievalRoute.route} · ${retrievalRoute.count || 0} 条`"
              >
                <p class="mb-2 whitespace-pre-wrap text-xs text-slate-500">Query：{{ retrievalRoute.query }}</p>
                <div class="space-y-2">
                  <div
                    v-for="item in retrievalRoute.items || []"
                    :key="`${retrievalRoute.route}-${item.file_id}-${item.chunk_id}`"
                    class="rounded-md border border-slate-200 bg-white p-2 text-xs"
                  >
                    <p class="font-medium text-slate-700">{{ item.file_name || '未命名资料' }} #{{ item.chunk_id }}</p>
                    <p class="mt-1 whitespace-pre-wrap text-slate-500">{{ item.excerpt }}</p>
                  </div>
                </div>
              </el-collapse-item>
            </el-collapse>
            <el-empty v-else description="暂无检索明细" />
          </el-tab-pane>

          <el-tab-pane label="记忆管理" name="memory">
            <div class="space-y-3 text-xs leading-5 text-slate-600">
              <p>滑动窗口：最近 4 轮；滑出窗口的完整问答轮次会合并进长期记忆；短期记忆超出 MEMORY_RECENT_MAX_CHARS 时压缩；长期记忆超出 MEMORY_SUMMARY_MAX_CHARS 时二次摘要。</p>
              <div v-if="memoryFocusVariables.length" class="grid gap-3 lg:grid-cols-2">
                <div
                  v-for="variable in memoryFocusVariables"
                  :key="variable.key"
                  class="rounded-md border border-brand-100 bg-brand-50/40 p-3"
                >
                  <div class="flex items-start justify-between gap-3">
                    <div>
                      <p class="text-xs font-medium text-slate-500">{{ variable.label }}</p>
                      <p class="mt-0.5 font-mono text-[13px] font-semibold text-slate-900">{{ variable.name }}</p>
                    </div>
                    <span class="rounded bg-white px-2 py-0.5 text-[11px] text-slate-500">
                      {{ memoryValueSize(variable.value) }}
                    </span>
                  </div>
                  <p class="mt-2 text-slate-500">{{ variable.description }}</p>
                  <pre class="mt-3 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-white p-3 text-[12px] leading-5 text-slate-700">{{ formatMemoryValue(variable.value) }}</pre>
                </div>
              </div>
              <div
                v-for="variable in memoryVariables"
                :key="variable.key"
                class="rounded-md border border-slate-200 bg-white p-3"
              >
                <p class="font-mono text-[13px] font-semibold text-slate-800">{{ variable.name }}</p>
                <p class="mt-1 text-slate-500">{{ variable.description }}</p>
                <pre class="mt-2 max-h-56 overflow-auto whitespace-pre-wrap rounded bg-slate-50 p-2">{{ formatMemoryValue(variable.value) }}</pre>
              </div>
              <el-empty v-if="!memoryVariables.length && !memoryFocusVariables.length" description="暂无记忆变量" />
            </div>
          </el-tab-pane>

          <el-tab-pane label="RAGAS评估" name="ragas">
            <div class="space-y-3 text-xs leading-5 text-slate-600">
              <div
                v-for="event in ragasEvents"
                :key="event.index"
                class="rounded-md border border-slate-200 bg-white p-3"
              >
                <p class="font-medium text-slate-800">{{ event.stage }}</p>
                <p class="mt-1">{{ event.note }}</p>
                <pre class="mt-2 max-h-56 overflow-auto rounded bg-slate-50 p-2">{{ formatJson(traceEventPayload(event)) }}</pre>
              </div>
              <el-empty v-if="!ragasEvents.length" description="暂无 RAGAS 事件" />
            </div>
          </el-tab-pane>

          <el-tab-pane label="原始JSON" name="json">
            <pre class="max-h-[70vh] overflow-auto rounded-md bg-slate-950 p-3 text-xs leading-5 text-slate-100">{{ prettyTrace }}</pre>
          </el-tab-pane>
        </el-tabs>
      </div>
    </el-drawer>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  Bottom,
  ChatDotSquare,
  CloseBold,
  CopyDocument,
  Delete,
  Document,
  Download,
  Edit,
  Loading,
  MagicStick,
  Picture,
  Plus,
  Refresh,
  Top,
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import MarkdownRenderer from '@/components/MarkdownRenderer.vue'
import TraceVariableFlow from '@/components/TraceVariableFlow.vue'
import { useChatStore } from '@/stores/chat'
import { useKnowledgeStore } from '@/stores/knowledge'
import { chatAPI } from '@/api/chat'
import { CHAT_SUGGESTIONS } from '@/utils/chatSuggestions'
import { copyText } from '@/utils/clipboard'
import { confirmCenteredDelete } from '@/utils/confirm'
import { getConversationExportErrorMessage, saveConversationMarkdown } from '@/utils/chatExport'
import { ACCEPTED_IMAGE_INPUT, validateImageFile } from '@/utils/fileValidation'
import {
  imageAnalysisWarningText as getImageAnalysisWarningText,
  shouldShowImageAnalysisWarning as isImageAnalysisWarningStatus,
} from '@/utils/imageAnalysisStatus'
import {
  MEMORY_VARIABLE_DESCRIPTIONS,
  buildMemoryVariables,
  filterTraceEventsByStageText,
  findRetrievalRoutes,
  findLatestMemoryValue,
  formatMemoryValue,
  memoryValueSize,
  normalizeTrace,
  traceEventPayload,
} from '@/utils/memoryTrace'
import { RAGAS_METRICS, RAGAS_STATUS, ragasStatusClass, ragasStatusText } from '@/utils/ragasStatus'
import { formatJson, formatScore } from '@/utils'

const route = useRoute()
const router = useRouter()
const chatStore = useChatStore()
const knowledgeStore = useKnowledgeStore()

const chatRef = ref(null)
const question = ref('')
const renaming = ref(false)
const renameTitle = ref('')
const renameRef = ref(null)
const sourcesVisible = ref(false)
const activeSources = ref([])
const traceVisible = ref(false)
const traceLoading = ref(false)
const traceTab = ref('timeline')
const activeTrace = ref({ trace_id: '', status: '', events: [] })
const attachments = ref([])
const uploadingAttachment = ref(false)
const exporting = ref(false)
const knowledgeBases = computed(() => knowledgeStore.knowledgeBases)
const selectedKnowledgeBaseId = ref(null)

const isCurrentConversationStreaming = computed(() => {
  if (!chatStore.streaming) return false
  if (!chatStore.streamingConversationId) return !route.params.id && !chatStore.currentId
  return chatStore.streamingConversationId === chatStore.currentId
})

// 切换会话的加载在途期间只做视觉承接：主区盖加载遮罩，空态与旧会话内容都不渲染。
const isConversationLoading = computed(() => chatStore.loadingConversationId !== null)

const currentKnowledgeBaseName = computed(() => {
  if (chatStore.currentConversation?.knowledge_base_name) {
    return chatStore.currentConversation.knowledge_base_name
  }
  const selectedBase = knowledgeBases.value.find((item) => item.id === selectedKnowledgeBaseId.value)
  return selectedBase?.name || '未选择知识库'
})

const traceEvents = computed(() => activeTrace.value?.events || [])

const retrievalRoutes = computed(() => findRetrievalRoutes(traceEvents.value))

const memoryEvents = computed(() =>
  filterTraceEventsByStageText(traceEvents.value, 'memory')
)

const memoryVariables = computed(() =>
  memoryEvents.value
    .flatMap((event) => buildMemoryVariables(event))
    .filter((variable) => !['recent_text', 'conv.memory_summary'].includes(variable.name))
)

const memoryFocusVariables = computed(() => {
  const variables = memoryEvents.value.flatMap((event) => buildMemoryVariables(event))
  return [
    {
      key: 'short-term-recent-text',
      label: '短期记忆',
      name: 'recent_text',
      description: MEMORY_VARIABLE_DESCRIPTIONS.recent_text,
      value: findLatestMemoryValue(variables, 'recent_text'),
    },
    {
      key: 'long-term-memory-summary',
      label: '长期记忆',
      name: 'conv.memory_summary',
      description: MEMORY_VARIABLE_DESCRIPTIONS['conv.memory_summary'],
      value: findLatestMemoryValue(variables, 'conv.memory_summary'),
    },
  ]
})

const ragasEvents = computed(() =>
  filterTraceEventsByStageText(traceEvents.value, 'ragas')
)

const prettyTrace = computed(() => formatJson(activeTrace.value))

// 路由 watcher 是会话加载的唯一入口（侧栏点击只负责导航）。这次加载的世代号
// 用来丢弃「已经切到别的会话之后才失败」的旧加载，避免它把路由回滚到错误的会话。
let routeLoadSeq = 0
let conversationsBootstrap = null
let conversationsLoaded = false

// 加载会话前先确保会话列表就绪：selectConversation 成功后要按列表里的
// knowledge_base_id 选中知识库。失败时下次导航重试。
function ensureConversationsLoaded() {
  if (conversationsLoaded) return Promise.resolve()
  if (!conversationsBootstrap) {
    conversationsBootstrap = chatStore
      .fetchConversations()
      .then(() => {
        conversationsLoaded = true
      })
      .catch(() => {})
      .finally(() => {
        conversationsBootstrap = null
      })
  }
  return conversationsBootstrap
}

watch(
  () => route.params.id,
  async (id) => {
    const seq = ++routeLoadSeq
    await refreshKnowledgeBases()
    if (seq !== routeLoadSeq) return
    await ensureConversationsLoaded()
    if (seq !== routeLoadSeq) return

    if (!id) {
      if (!chatStore.streaming) {
        chatStore.clearMessages()
      } else {
        // 流式回答期间回到空白页不清视图，但同样要放弃在途的会话加载，
        // 否则它晚到的响应会写进这个已经没有会话的页面。
        chatStore.cancelConversationLoad()
      }
      syncSelectedKnowledgeBase()
      return
    }
    // 路由指向的正是已加载的会话：不再重复加载，只需作废在途的其它加载。
    if (id === chatStore.currentId) {
      chatStore.cancelConversationLoad()
      syncSelectedKnowledgeBase()
      return
    }

    // 失败时回滚到发起这次切换前仍完整显示着的会话。
    const previousId = chatStore.currentId
    try {
      await chatStore.selectConversation(id)
    } catch {
      // 快速切换时旧加载可能晚失败：新的那次已经接管，不要回滚它的路由。
      if (seq !== routeLoadSeq) return
      ElMessage.error('对话加载失败')
      router.replace(previousId ? `/chat/${previousId}` : '/chat')
      return
    }
    if (seq !== routeLoadSeq) return
    // selectConversation 可能成功返回却没提交这个 id：加载在飞时流已经认领了
    // 新会话（currentId 被改写），store 按发起时的 currentId 作废了这次加载。
    // 此时 URL 还停在旧会话上，界面与刷新都会把用户带回旧会话，必须校正到
    // 实际生效的会话。正常加载时 activeId === id，不会导航。
    const activeId = chatStore.currentId
    if (activeId && activeId !== id) {
      router.replace(`/chat/${activeId}`)
    }
    syncSelectedKnowledgeBase()
  },
  { immediate: true }
)

watch(
  () => chatStore.streaming,
  (value) => {
    if (!value) {
      scrollToBottom()
    }
  }
)

watch(
  () => chatStore.currentId,
  (id) => {
    if (id && !route.params.id && !chatStore.streaming) {
      router.replace(`/chat/${id}`)
    }
  }
)

watch(
  () => chatStore.pendingRouteConversationId,
  (id) => {
    if (id && !route.params.id) {
      router.replace(`/chat/${id}`)
    }
  }
)

watch(
  () => chatStore.errorMessage,
  (message) => {
    if (message) {
      ElMessage.error(message)
    }
  }
)

function askSuggestion(text) {
  question.value = text
  sendMessage()
}

function createNewConversation() {
  chatStore.clearMessages()
  router.push('/chat')
  syncSelectedKnowledgeBase()
}

async function exportConversation() {
  if (!chatStore.currentId || chatStore.streaming || exporting.value) return

  // 等待下载时允许切换会话，文件始终对应这次点击的会话。
  const conversationId = chatStore.currentId
  const title = chatStore.currentConversation?.title || '未命名对话'
  exporting.value = true
  try {
    const response = await chatAPI.exportConversation(conversationId)
    saveConversationMarkdown(response, title)
  } catch (error) {
    ElMessage.error(await getConversationExportErrorMessage(error))
  } finally {
    exporting.value = false
  }
}

function sendMessage() {
  const text = question.value.trim()
  if ((!text && !attachments.value.length) || chatStore.streaming) return

  const sendingAttachments = [...attachments.value]
  question.value = ''
  attachments.value = []
  chatStore.sendMessage(text, sendingAttachments)
  scrollToBottom()
}

function handleInputEnter(event) {
  if (event.isComposing || event.keyCode === 229) return
  if (event.shiftKey) return
  event.preventDefault()
  sendMessage()
}

function stopGeneration() {
  chatStore.stopGeneration()
}

async function copyMessage(content) {
  await copyText(content)
}

function onFeedback(message, value) {
  // 再点同向 = 取消（归 0）：0 是"未反馈"的唯一表示，保证选中态是二元的。
  const next = message.feedback === value ? 0 : value
  // 失败提示由 request.js 拦截器统一给出；这里必须接住 rejection（store 回滚后会重新抛出）。
  chatStore.setMessageFeedback(message.id, next).catch(() => {})
}

async function onDeleteMessage(message) {
  if (!message.id || chatStore.streaming) return

  const isQuestion = message.role === 'user'
  try {
    await confirmCenteredDelete(
      isQuestion ? '删除该提问及其后的全部消息？' : '删除这条回答？',
      '删除消息'
    )
  } catch {
    // 取消确认：不调用删除接口。
    return
  }

  // 确认框打开的这段时间里用户可能已经开始生成回答：store 此时直接返回 false、
  // 不发删除请求。不能沿用「调用过就成功」的写法，否则消息还在却提示已删除。
  const conversationId = chatStore.currentId
  const wasLast = chatStore.messages[chatStore.messages.length - 1]?.id === message.id
  try {
    const deleted = await chatStore.deleteMessage(message.id)
    if (!deleted) {
      ElMessage.warning('正在生成回答，暂不能删除')
      return
    }
    ElMessage.success(isQuestion ? '已删除该提问及其后的消息' : '已删除该回答')
    // 请求期间切走时不要滚动另一个会话的窗口。
    if (chatStore.currentId === conversationId && (isQuestion || wasLast)) scrollToBottom()
  } catch {
    // 失败提示由 request.js 拦截器统一给出，这里不再弹第二次。
  }
}

async function regenerate(index) {
  const messages = chatStore.messages
  const previousUserMessage = [...messages]
    .slice(0, index)
    .reverse()
    .find((message) => message.role === 'user')

  if (!previousUserMessage || chatStore.streaming) return

  chatStore.replaceMessages(messages.slice(0, index))
  question.value = previousUserMessage.content
  await sendMessage()
}

function startRename() {
  renaming.value = true
  renameTitle.value = chatStore.currentConversation?.title || ''
  nextTick(() => renameRef.value?.focus())
}

async function confirmRename() {
  if (!renaming.value) return

  renaming.value = false
  const title = renameTitle.value.trim()
  if (!title || title === chatStore.currentConversation?.title) return

  try {
    await chatStore.renameConversation(chatStore.currentId, title)
    ElMessage.success('已重命名')
  } catch {
    ElMessage.error('重命名失败')
  }
}

function scrollToBottom() {
  nextTick(() => {
    chatRef.value?.scrollTo({ top: chatRef.value.scrollHeight, behavior: 'smooth' })
  })
}

// 向前翻页会把更早的消息插到列表头部，这里按插入高度补回 scrollTop，
// 否则用户点一次「加载更早的消息」就会被弹到那批老消息的开头。
async function loadOlderMessages() {
  const container = chatRef.value
  const previousHeight = container?.scrollHeight ?? 0
  const previousTop = container?.scrollTop ?? 0
  try {
    await chatStore.loadOlderMessages()
  } catch {
    // 请求失败已经由 axios 拦截器提示，这里只保证不把异常抛成未处理的拒绝。
  }
  await nextTick()
  if (container) {
    container.scrollTop = container.scrollHeight - previousHeight + previousTop
  }
}

function openSources(sources) {
  activeSources.value = sources || []
  sourcesVisible.value = true
}

function hasDisplayValue(value) {
  return value !== null && value !== undefined && value !== ''
}

function sourceText(source) {
  if (hasDisplayValue(source?.content)) return source.content
  if (hasDisplayValue(source?.excerpt)) return source.excerpt
  return ''
}

// 来源卡片的 file_id 是跨页跳转的唯一凭据：只有正整数才是可寻址的文件 id。
// 缺失、0、非数字一律返回 null —— 卡片保持原来的静态展示，不给出可点击的样子（#255）。
function sourceFileId(source) {
  const raw = source?.file_id
  const text = typeof raw === 'number' ? String(raw) : raw
  if (typeof text !== 'string' || !/^\d+$/.test(text)) return null
  const fileId = Number(text)
  return Number.isSafeInteger(fileId) && fileId > 0 ? fileId : null
}

function openSourceFile(source) {
  const fileId = sourceFileId(source)
  if (!fileId) return
  // 先收起抽屉再跳转：离开会话页之后参考资料不应残留为“还开着”的状态。
  sourcesVisible.value = false
  router.push({ path: '/knowledge', query: { file: String(fileId) } })
}

function hasTrace(message) {
  const trace = message?.learning_trace || message?.retrieval_trace?.learning_trace || {}
  return Boolean(message?.trace_id || trace.trace_id || trace.events?.length)
}

async function openTrace(message) {
  traceVisible.value = true
  traceTab.value = 'timeline'
  const embeddedTrace = message?.learning_trace || message?.retrieval_trace?.learning_trace || {}
  activeTrace.value = embeddedTrace || { trace_id: '', status: '', events: [] }
  const traceId = message?.trace_id || embeddedTrace.trace_id
  if (!traceId && !message?.id) return

  traceLoading.value = true
  try {
    const response = message?.id
      ? await chatAPI.getMessageTrace(message.id)
      : await chatAPI.getTrace(traceId)
    activeTrace.value = normalizeTrace(response)
  } catch {
    if (!activeTrace.value?.events?.length) {
      ElMessage.warning('暂无可加载的流程详情')
    }
  } finally {
    traceLoading.value = false
  }
}

function shouldShowImageAnalysisWarning(message) {
  return isImageAnalysisWarningStatus(message?.image_analysis_status)
}

function imageAnalysisWarningText(message) {
  return getImageAnalysisWarningText(
    message?.image_analysis_status,
    message?.image_analysis_error
  )
}

async function refreshKnowledgeBases() {
  await knowledgeStore.refreshKnowledgeBases().catch(() => [])
  syncSelectedKnowledgeBase()
}

function syncSelectedKnowledgeBase() {
  const preferredId = chatStore.selectedKnowledgeBaseId
  const hasPreferred = knowledgeBases.value.some((item) => item.id === preferredId)
  const nextId = hasPreferred ? preferredId : knowledgeBases.value[0]?.id || null
  selectedKnowledgeBaseId.value = nextId
  chatStore.setSelectedKnowledgeBaseId(nextId)
}

function changeKnowledgeBase(id) {
  chatStore.setSelectedKnowledgeBaseId(id)
  syncSelectedKnowledgeBase()
}

async function handleImageUpload(file) {
  const validationMessage = validateImageFile(file, {
    maxSizeMb: 5,
    typeMessage: '仅支持 png、jpg、jpeg、webp 图片',
    sizeMessage: '图片不能超过 5MB',
  })
  if (validationMessage) {
    ElMessage.error(validationMessage)
    return false
  }

  uploadingAttachment.value = true
  try {
    const response = await chatAPI.uploadAttachment(file)
    attachments.value.push(response)
    ElMessage.success('图片已添加')
  } finally {
    uploadingAttachment.value = false
  }
  return false
}

function removeAttachment(attachment) {
  attachments.value = attachments.value.filter((item) => item !== attachment)
}
</script>
