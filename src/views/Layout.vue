<template>
  <el-container class="h-screen bg-slate-100">
    <el-aside width="280px" class="flex flex-col border-r border-slate-200 bg-white">
      <div class="flex h-16 items-center gap-3 border-b border-slate-200 px-5">
        <div class="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600 text-white">
          <el-icon><Connection /></el-icon>
        </div>
        <div>
          <div class="text-sm font-semibold leading-5 text-slate-900">企业知识库</div>
          <div class="text-xs text-slate-500">智能问答 Agent</div>
        </div>
      </div>

      <el-menu :default-active="activeMenu" :router="true" class="border-none py-3">
        <el-menu-item index="/chat">
          <el-icon><ChatDotSquare /></el-icon>
          <span>智能问答</span>
        </el-menu-item>
        <el-menu-item index="/knowledge">
          <el-icon><FolderOpened /></el-icon>
          <span>知识库管理</span>
        </el-menu-item>
      </el-menu>

      <div class="flex min-h-0 flex-1 flex-col border-t border-slate-200">
        <div class="flex items-center justify-between px-4 py-3">
          <span class="text-xs font-medium text-slate-500">历史对话</span>
          <div class="flex items-center gap-2">
            <el-tooltip :content="chatStore.historyManageMode ? '完成管理' : '管理对话'" placement="right">
              <el-button
                size="small"
                circle
                :icon="Operation"
                :type="chatStore.historyManageMode ? 'primary' : 'default'"
                @click="toggleHistoryManageMode"
              />
            </el-tooltip>
            <el-tooltip content="新建对话" placement="right">
              <el-button size="small" circle :icon="Plus" @click="newConversation" />
            </el-tooltip>
          </div>
        </div>

        <div
          v-if="chatStore.historyManageMode"
          class="mx-3 mb-3 rounded-lg border border-slate-200 bg-slate-50 px-3 py-3"
        >
          <div class="flex items-center justify-between gap-3">
            <button
              type="button"
              class="text-xs font-medium text-brand-600 transition-colors hover:text-brand-700"
              @click="chatStore.toggleSelectAllConversations"
            >
              {{ allConversationsSelected ? '取消全选' : '全选' }}
            </button>
            <span class="text-xs text-slate-400">
              已选 {{ chatStore.selectedConversationIds.length }} / {{ chatStore.conversations.length }}
            </span>
          </div>
          <div class="mt-3 flex items-center gap-2">
            <el-button
              size="small"
              type="danger"
              :disabled="!chatStore.selectedConversationIds.length"
              @click="removeSelectedConversations"
            >
              删除选中
            </el-button>
            <el-button size="small" @click="chatStore.exitHistoryManageMode()">完成</el-button>
          </div>
        </div>

        <div v-loading="chatStore.loading" class="flex-1 min-h-0 overflow-y-auto px-2 pb-3">
          <button
            v-for="conversation in chatStore.conversations"
            :key="conversation.id"
            type="button"
            :class="[
              'group flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm transition-colors',
              chatStore.isConversationSelected(conversation.id)
                ? 'bg-brand-50 text-brand-700 ring-1 ring-brand-200'
                : conversation.id === chatStore.currentId
                ? 'bg-brand-50 text-brand-700'
                : 'text-slate-600 hover:bg-slate-100',
            ]"
            @click="handleConversationClick(conversation.id)"
          >
            <el-checkbox
              v-if="chatStore.historyManageMode"
              :model-value="chatStore.isConversationSelected(conversation.id)"
              @click.stop
              @change="chatStore.toggleConversationSelection(conversation.id)"
            />
            <el-icon :size="15"><ChatLineRound /></el-icon>
            <!-- 行内改名（issue #257）：编辑态把标题换成输入框，回车提交 / Esc 取消 /
                 失焦取消（与 Chat.vue 的发送框相反，那里失焦是提交）。 -->
            <el-input
              v-if="editingConversationId === conversation.id"
              ref="renameInputRef"
              v-model.trim="editingTitle"
              size="small"
              maxlength="40"
              @keyup.enter="submitRename(conversation, $event)"
              @keyup.esc="cancelRename()"
              @blur="cancelRename()"
              @click.stop
            />
            <span v-else class="flex-1 truncate">{{ conversation.title || '未命名对话' }}</span>
            <el-button
              v-if="!chatStore.historyManageMode"
              link
              size="small"
              :icon="Edit"
              class="rename-entry opacity-0 group-hover:opacity-100"
              @click.stop="startRename(conversation)"
            />
            <el-popconfirm
              v-if="!chatStore.historyManageMode"
              title="确定删除该对话吗？"
              confirm-button-text="删除"
              cancel-button-text="取消"
              confirm-button-type="danger"
              placement="right"
              width="190"
              @confirm="removeConversation(conversation.id)"
            >
              <template #reference>
                <el-button
                  link
                  type="danger"
                  size="small"
                  :icon="Delete"
                  class="opacity-0 group-hover:opacity-100"
                  @click.stop
                />
              </template>
            </el-popconfirm>
          </button>

          <!-- 会话列表有页大小上限（issue #191）：更早的会话按需往后翻。
               没有这个入口时，超出第一页的会话在侧栏里就是静默消失。 -->
          <button
            v-if="chatStore.hasMoreConversations"
            type="button"
            :disabled="chatStore.loadingMoreConversations"
            class="mt-1 flex w-full items-center justify-center rounded-md px-3 py-2 text-xs font-medium text-brand-600 transition-colors hover:bg-brand-50 disabled:cursor-not-allowed disabled:text-slate-400"
            @click="chatStore.loadMoreConversations()"
          >
            {{ chatStore.loadingMoreConversations ? '加载中…' : '加载更早的对话' }}
          </button>

          <div
            v-if="!chatStore.conversations.length && !chatStore.loading"
            class="px-4 py-8 text-center text-xs text-slate-400"
          >
            暂无历史对话
          </div>
        </div>
      </div>

      <div class="border-t border-slate-200 p-4">
        <el-dropdown trigger="click" @command="handleUserCommand">
          <button type="button" class="flex w-full items-center gap-3 rounded-md p-2 text-left hover:bg-slate-100">
            <el-avatar :size="32" :src="avatarSrc">{{ userStore.avatarText }}</el-avatar>
            <span class="min-w-0 flex-1">
              <span class="block truncate text-sm font-medium text-slate-800">{{ userStore.displayName }}</span>
              <span class="block text-xs text-slate-500">工作台账户</span>
            </span>
            <el-icon class="text-slate-400"><ArrowDown /></el-icon>
          </button>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="profile">个人设置</el-dropdown-item>
              <el-dropdown-item divided command="logout">退出登录</el-dropdown-item>
            </el-dropdown-menu>
          </template>
        </el-dropdown>
      </div>
    </el-aside>

    <el-container class="min-w-0">
      <el-main class="min-w-0 p-0">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import {
  ArrowDown,
  ChatDotSquare,
  ChatLineRound,
  Connection,
  Delete,
  Edit,
  FolderOpened,
  Operation,
  Plus,
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import { useChatStore } from '@/stores/chat'
import { useUserStore } from '@/stores/user'
import { confirmCenteredDelete } from '@/utils/confirm'

const route = useRoute()
const router = useRouter()
const chatStore = useChatStore()
const userStore = useUserStore()

const activeMenu = computed(() => {
  if (route.path.startsWith('/knowledge')) return '/knowledge'
  return '/chat'
})
// 直接取 store 里那一个（本地头像是带 token 取回后现做的 object URL，见 stores/user.js）。
const avatarSrc = computed(() => userStore.avatarSrc)
const allConversationsSelected = computed(() =>
  Boolean(chatStore.conversations.length) &&
  chatStore.selectedConversationIds.length === chatStore.conversations.length
)

// 行内改名（issue #257）：同一时刻至多一行处于编辑态，null 表示没有。
const editingConversationId = ref(null)
const editingTitle = ref('')
const renameInputRef = ref(null)

onMounted(() => {
  userStore.fetchProfile().catch(() => {})
  chatStore.fetchConversations()
})

function focusRenameInput() {
  // `ref` 写在 v-for 里时 Vue 会带上 ref_for，落下来的是数组而不是单个实例；
  // 同一时刻至多一个编辑框，取末尾那个即可。
  const target = renameInputRef.value
  const input = Array.isArray(target) ? target[target.length - 1] : target
  input?.focus()
}

function startRename(conversation) {
  // 换行编辑前先收掉上一行 —— 走的是同一个「退出编辑态」出口。
  cancelRename()
  editingConversationId.value = conversation.id
  editingTitle.value = conversation.title || ''
  nextTick(focusRenameInput)
}

async function submitRename(conversation, event) {
  // 输入法组字中的回车是「上屏」不是「提交」（同 Chat.vue 发送框的守卫）。
  if (event.isComposing || event.keyCode === 229) return

  const id = conversation.id
  const next = editingTitle.value.trim()
  // 先同步退出编辑态、再 await：提交与失焦是两条会互相追尾的路径，
  // 后到的那一方会被 cancelRename 的 null 挡掉。
  editingConversationId.value = null
  editingTitle.value = ''
  if (!next || next === conversation.title) return

  try {
    await chatStore.renameConversation(id, next)
    ElMessage.success('已重命名')
  } catch {
    // await 在赋值之前，失败的标题不会写进列表，这里只需退出编辑态并告知。
    ElMessage.error('重命名失败')
  }
}

function cancelRename() {
  if (editingConversationId.value === null) return
  editingConversationId.value = null
  editingTitle.value = ''
}

function newConversation() {
  cancelRename()
  chatStore.exitHistoryManageMode()
  chatStore.clearMessages()
  router.push('/chat')
}

function selectConversation(id) {
  chatStore.exitHistoryManageMode()
  // 只负责导航：消息加载由 Chat.vue 的路由 watcher 统一处理，
  // 两处各发一次请求会让「失败回滚」拿到两份互相追逐的结果。
  router.push(`/chat/${id}`)
}

function handleConversationClick(id) {
  // 正在改名的这一行：点在编辑框上的事件已经被 @click.stop 挡住，这里是兜底 ——
  // 不允许「正在编辑的行」被这一下顺手选中/跳转。
  if (editingConversationId.value === id) return
  if (chatStore.historyManageMode) {
    chatStore.toggleConversationSelection(id)
    return
  }
  selectConversation(id)
}

async function removeConversation(id) {
  await chatStore.deleteConversation(id)
  if (route.params.id === id) {
    router.push('/chat')
  }
}

function toggleHistoryManageMode() {
  // 管理模式每行会换成复选框、改名入口随之收起，编辑态在这里先收干净。
  cancelRename()
  if (chatStore.historyManageMode) {
    chatStore.exitHistoryManageMode()
    return
  }
  chatStore.enterHistoryManageMode()
}

async function removeSelectedConversations() {
  if (!chatStore.selectedConversationIds.length) return

  await confirmCenteredDelete(
    `确定删除选中的 ${chatStore.selectedConversationIds.length} 条历史对话吗？删除后不可恢复。`,
    '批量删除对话'
  )

  const deletingIds = [...chatStore.selectedConversationIds]
  for (const id of deletingIds) {
    await chatStore.deleteConversation(id)
    if (route.params.id === id) {
      router.push('/chat')
    }
  }
  chatStore.exitHistoryManageMode()
}

async function handleUserCommand(command) {
  if (command === 'profile') {
    router.push('/profile')
    return
  }

  if (command === 'logout') {
    await userStore.logout()
    chatStore.clearMessages()
    chatStore.exitHistoryManageMode()
    router.replace('/login')
  }
}
</script>
