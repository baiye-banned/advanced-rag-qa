// Layout.vue 的挂载用例：真 mount 外壳视图，覆盖「布局切换」主路径（issue #180 验收 2）。
//
// 由来：`git grep Layout.vue tests/` 此前**零命中** —— 这个视图（侧栏菜单、历史对话列、
// 账户区、`<router-view/>`）一行执行证据都没有。它是所有工作台页面的外壳，
// 却因为「看起来只是排版」而落在覆盖缺口里。
//
// 与既有用例的分工：
//   tests/chatStore.test.js  测 store 的会话状态机（进/出管理模式、选中、全选）
//   本文件                    视图真的跑起来之后：菜单跟着路由走、会话列渲染的是 store 的值、
//                            账户区显示的是 user store 的名字
//
// 替身只扎在 `api/request.js`（HTTP 出口）。chat/user 两个 pinia store、vue-router、
// Element Plus 的 el-menu 全是真货 —— 所以「布局切换」是真点菜单、真导航、真重算
// activeMenu，而不是断言一个被 stub 的开关。
//
// vue-router 必须动态 import（它 import 'vue'，见 chatViewMount.test.js 里的同一条说明）。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { calls, resetRequestStub, respond } from './helpers/stubApiRequest.js'

const { createMemoryHistory, createRouter } = await import('vue-router')

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
}

const CONV1 = { id: 'c1', title: '第一段对话' }
const CONV2 = { id: 'c2', title: '第二段对话' }
const PROFILE = { username: '张三', avatar: '', created_at: '2024-01-01T00:00:00Z' }

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
      { path: '/knowledge', name: 'Knowledge', component: blank },
      { path: '/profile', name: 'UserProfile', component: blank },
      { path: '/login', name: 'Login', component: blank },
    ],
  })
}

async function mountLayout({ conversations = [CONV1, CONV2], route = '/chat' } = {}) {
  resetRequestStub()
  // 每个用例拿到一份副本：store 的 renameConversation 是**就地改写**标题
  // （src/stores/chat.js 里 `conversation.title = title`），共用同一批对象会让
  // 上一个用例改过的名字漏进下一个（CONV1 是模块级常量，一改全脏）。
  const seed = conversations.map((conversation) => ({ ...conversation }))
  // 有 token 才会真的去拉 profile（useUserStore.fetchProfile 在无 token 时直接返回）。
  localStorage.setItem('token', 'test-token')
  localStorage.setItem('username', '张三')

  respond('get', (url) => {
    if (url === '/chat/conversations') return seed
    if (url === '/user/profile') return PROFILE
    return {}
  })

  const router = createTestRouter()
  await router.push(route)
  await router.isReady()

  const view = await mountSfc('views/Layout.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  return { view, router }
}

// 侧栏菜单项：文案与 :index 一一对应（模板 :15-22）。
function menuItem(view, label) {
  return view
    .queryAll('.el-menu-item')
    .find((node) => node.textContent.trim() === label)
}

function activeMenuItems(view) {
  return view
    .queryAll('.el-menu-item.is-active')
    .map((node) => node.textContent.trim())
}

async function close(view) {
  view.elementPlusUnpatch()
  await view.unmount()
  localStorage.clear()
}

// ---------------------------------------------------------------------------
// a. 主路径：布局切换（点菜单 -> 路由跳转 -> 激活态跟着走）
// ---------------------------------------------------------------------------

test('布局切换：点「知识库管理」菜单项跳转到 /knowledge，激活态随之切换', async () => {
  const { view, router } = await mountLayout({ route: '/chat' })

  try {
    assert.deepEqual(activeMenuItems(view), ['智能问答'], '进入 /chat 时「智能问答」应当处于激活态')

    const item = menuItem(view, '知识库管理')
    assert.ok(item, '侧栏应当渲染出「知识库管理」菜单项')
    item.click()
    await until(view, () => router.currentRoute.value.path === '/knowledge')
    await settle(view)

    // 承重：路由真的被推走了（`el-menu` 的 `:router="true"` 接线），
    // 且 `activeMenu` computed 跟着 route.path 重算 —— 少了它，点了菜单高亮还停在原处。
    assert.equal(router.currentRoute.value.path, '/knowledge', '点菜单项应当把路由推到 /knowledge')
    assert.deepEqual(activeMenuItems(view), ['知识库管理'], '激活态应当跟着路由切换')

    // 切回来同样成立（断言不是「只在新路径上碰巧成立」）。
    menuItem(view, '智能问答').click()
    await until(view, () => router.currentRoute.value.path === '/chat')
    await settle(view)
    assert.deepEqual(activeMenuItems(view), ['智能问答'], '切回 /chat 时激活态应当切回来')

    // 上面两条都经由「点菜单」——而 `el-menu` 在被点击时自己也会更新内部高亮，
    // 所以单靠点击区分不出 `activeMenu` 到底有没有跟着 `route.path` 重算。
    // 这条**不碰菜单**、直接推路由：只有 computed 真的重算，高亮才会跟上。
    await router.push('/knowledge')
    await settle(view)
    assert.deepEqual(
      activeMenuItems(view),
      ['知识库管理'],
      '不经菜单、直接改路由时激活态也应当跟上（这才钉住 activeMenu 跟着 route.path）',
    )
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// b. 外壳数据：历史对话列与账户区渲染的是两个 store 的真实值
// ---------------------------------------------------------------------------

test('挂载：历史对话列与账户区渲染的是 store 的真实值', async () => {
  const { view } = await mountLayout()

  try {
    // 挂载时真的拉了会话列表与用户资料（壳自己 onMounted 里的两件事）。
    assert.ok(
      calls.some((entry) => entry.method === 'get' && entry.args[0] === '/chat/conversations'),
      '挂载时应当拉一次历史对话'
    )
    assert.ok(
      calls.some((entry) => entry.method === 'get' && entry.args[0] === '/user/profile'),
      '挂载时应当拉一次用户资料'
    )

    assert.match(view.text(), /第一段对话/)
    assert.match(view.text(), /第二段对话/)
    // 账户区：displayName / avatarText 都来自 user store（profile 回来之后）。
    assert.match(view.text(), /张三/)
    assert.match(view.text(), /企业知识库/, '侧栏标题应当渲染')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// c. 交互：管理模式开关与「新建对话」都把壳的状态改对了
// ---------------------------------------------------------------------------

test('管理模式开关与「新建对话」：壳的接线真的改了 store 里的状态', async () => {
  const { view, router } = await mountLayout({ route: '/chat/c1' })

  try {
    // 进入管理模式：模板 :44 的管理区（全选 / 删除选中 / 完成）随之出现。
    // 「历史对话」标题行里的两个图标按钮：第一个是管理模式开关、第二个是新建对话。
    const circleButtons = view
      .queryAll('button')
      .filter((node) => node.classList.contains('is-circle'))
    assert.ok(circleButtons.length >= 2, '标题行应当渲染管理对话与新建对话两个按钮')

    circleButtons[0].click()
    await until(view, () => view.text().includes('全选'))
    assert.match(view.text(), /已选 0 \/ 2/, '进入管理模式后应当出现管理区并显示选中计数')

    // 「新建对话」：清空当前会话并把路由推回 /chat。
    circleButtons[1].click()
    await until(view, () => router.currentRoute.value.path === '/chat')
    assert.equal(router.currentRoute.value.path, '/chat', '点「新建对话」应当回到 /chat')
    const manageCleared = !view.text().includes('全选')
    assert.equal(manageCleared, true, '「新建对话」应当顺带退出管理模式')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// d. 重命名：侧栏会话行内改名（issue #257）
// ---------------------------------------------------------------------------
//
// 被测路径是三层接力：模板里的行内编辑框 -> store 的 renameConversation ->
// `PUT /chat/conversations/:id`（替身记在 calls 里）。
//
// 选择器约定：不用按钮下标（标题行的管理/新建按钮、以及管理模式里的复选框都会动到
// `queryAll('button')` 的下标），改用一个稳定的类名 `rename-entry`；
// 输入框一律限定在 `.conversation-list` 内 —— 侧栏顶部的标题检索框（issue #251）
// 也是 input，不限定的话 `query('input')` 会先拿到它。
//
// 事件一律真派发（`Event('input')` / `KeyboardEvent('keyup')` / `FocusEvent('blur')`），
// 不直接调组件内部方法——这样 v-model、`.stop`、按键修饰符都是真的在执行。

// 重命名入口：每行一个，顺序与会话行一致。
function renameIcons(view) {
  return view.queryAll('.rename-entry')
}

// 历史对话行（模板 :74 起，行根是 `<button class="group …">`）。
function conversationRows(view) {
  return view.queryAll('button.group')
}

function putCalls() {
  return calls.filter((entry) => entry.method === 'put')
}

// 会话列表请求（挂载时一次，之后检索 / 翻页各算一次）。
function conversationListCalls() {
  return calls.filter((entry) => entry.method === 'get' && entry.args[0] === '/chat/conversations')
}

async function openRename(view, index) {
  const icon = renameIcons(view)[index]
  assert.ok(icon, `第 ${index + 1} 行应当渲染出重命名入口`)
  icon.click()
  await settle(view)
  const input = view.query('.conversation-list input')
  assert.ok(input, '进入编辑态后应当渲染出输入框')
  return input
}

// v-model 靠 input 事件回填，直接改 `.value` 不派事件 store 侧是收不到的。
function typeInto(input, value) {
  input.value = value
  input.dispatchEvent(new Event('input', { bubbles: true }))
}

function pressKey(node, key, init = {}) {
  node.dispatchEvent(new KeyboardEvent('keyup', { key, bubbles: true, ...init }))
}

test('重命名 R1：点图标进入编辑态，输入框预填当前标题并自动聚焦', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)

    assert.equal(view.queryAll('.conversation-list input').length, 1, '同一时刻只应当有一个输入框')
    assert.equal(input.value, CONV1.title, '输入框应当预填当前标题')
    const focused = await until(view, () => document.activeElement === input)
    assert.ok(focused, '进入编辑态后输入框应当自动获得焦点')
  } finally {
    await close(view)
  }
})

test('重命名 R2：点另一行的图标只保留一个编辑框，并切到新行', async () => {
  const { view } = await mountLayout()

  try {
    const first = await openRename(view, 1)
    assert.equal(first.value, CONV2.title, '应当先进入第二行的编辑态')

    renameIcons(view)[0].click()
    await settle(view)

    const inputs = view.queryAll('.conversation-list input')
    assert.equal(inputs.length, 1, '切行之后仍应当只有一个输入框')
    assert.equal(inputs[0].value, CONV1.title, '编辑框应当切到新行并预填它的标题')
  } finally {
    await close(view)
  }
})

test('重命名 R3：回车提交，发一次 PUT、就地更新标题并提示成功', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)
    typeInto(input, '改过的标题')
    pressKey(input, 'Enter')
    await settle(view)

    const puts = putCalls()
    assert.equal(puts.length, 1, '提交应当只发一次 PUT')
    assert.equal(puts[0].args[0], '/chat/conversations/c1', '应当打到这条会话的端点')
    assert.deepEqual(puts[0].args[1], { title: '改过的标题' }, '请求体应当是 { title }')

    assert.match(view.text(), /改过的标题/, '列表应当显示新标题')
    assert.doesNotMatch(view.text(), /第一段对话/, '旧标题应当被替换')
    assert.ok(
      view.messages.some((entry) => entry.level === 'success' && entry.message === '已重命名'),
      '提交成功应当提示「已重命名」'
    )
    assert.equal(view.queryAll('.conversation-list input').length, 0, '提交之后应当退出编辑态')
  } finally {
    await close(view)
  }
})

test('重命名 R4：Esc 取消，不提交且保留原标题', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)
    typeInto(input, '不该保存的标题')
    pressKey(input, 'Escape')
    await settle(view)

    assert.equal(putCalls().length, 0, 'Esc 不应当发请求')
    assert.match(view.text(), /第一段对话/, '取消后应当保留原标题')
    assert.doesNotMatch(view.text(), /不该保存的标题/, '草稿不应当出现在列表里')
    assert.equal(view.queryAll('.conversation-list input').length, 0, '取消之后应当退出编辑态')
  } finally {
    await close(view)
  }
})

test('重命名 R5：失焦取消，不提交且保留原标题', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)
    typeInto(input, '失焦丢弃的标题')
    input.dispatchEvent(new FocusEvent('blur'))
    await settle(view)

    assert.equal(putCalls().length, 0, '失焦不应当发请求')
    assert.match(view.text(), /第一段对话/, '失焦取消后应当保留原标题')
    assert.equal(view.queryAll('.conversation-list input').length, 0, '失焦之后应当退出编辑态')
  } finally {
    await close(view)
  }
})

test('重命名 R6：输入框带 40 字上限', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)
    // jsdom 不实现 maxlength 的截断（程序化写 .value 多长就是多长），所以只能钉属性本身。
    assert.equal(input.getAttribute('maxlength'), '40', '输入框应当带 40 字上限')
  } finally {
    await close(view)
  }
})

test('重命名 R7：纯空白标题被拦截，不发请求', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)
    typeInto(input, '   ')
    pressKey(input, 'Enter')
    await settle(view)

    assert.equal(putCalls().length, 0, '纯空白不应当发请求')
    assert.match(view.text(), /第一段对话/, '列表里的标题不应当变化')
    assert.equal(view.queryAll('.conversation-list input').length, 0, '拦截之后仍然退出编辑态')
  } finally {
    await close(view)
  }
})

test('重命名 R8：与原标题相同则短路，不发请求', async () => {
  const { view } = await mountLayout()

  try {
    // 一个字都不改，直接回车。
    const input = await openRename(view, 0)
    pressKey(input, 'Enter')
    await settle(view)

    assert.equal(putCalls().length, 0, '同名不应当发请求')
    assert.match(view.text(), /第一段对话/, '列表里的标题不应当变化')
    assert.equal(view.queryAll('.conversation-list input').length, 0, '短路之后仍然退出编辑态')
  } finally {
    await close(view)
  }
})

test('重命名 R9：请求失败时保留原标题、退出编辑态并提示失败', async () => {
  const { view } = await mountLayout()

  try {
    respond('put', () => {
      throw new Error('boom')
    })

    const input = await openRename(view, 0)
    typeInto(input, '会失败的标题')
    pressKey(input, 'Enter')
    await settle(view)

    assert.equal(putCalls().length, 1, '应当尝试提交一次')
    assert.match(view.text(), /第一段对话/, '失败后应当保留原标题')
    assert.doesNotMatch(view.text(), /会失败的标题/, '失败的标题不应当写进列表')
    assert.ok(
      view.messages.some((entry) => entry.level === 'error' && entry.message === '重命名失败'),
      '提交失败应当提示「重命名失败」'
    )
    assert.equal(view.queryAll('.conversation-list input').length, 0, '失败之后应当退出编辑态')
  } finally {
    await close(view)
  }
})

test('重命名 R10：管理模式里不出现重命名入口，编辑态随之中止', async () => {
  const { view } = await mountLayout()

  try {
    await openRename(view, 0)

    // 标题行第一个圆形按钮是管理模式开关（模板 :29）。
    const circleButtons = view
      .queryAll('button')
      .filter((node) => node.classList.contains('is-circle'))
    circleButtons[0].click()
    await until(view, () => view.text().includes('全选'))
    await settle(view)

    assert.equal(renameIcons(view).length, 0, '管理模式不应当渲染重命名入口')
    // 增量断言：不写死输入框数量（管理模式本来就有每行一个复选框），
    // 只钉「除复选框之外没有多余的输入框」。
    const checkboxes = view.queryAll('.conversation-list .el-checkbox')
    assert.ok(checkboxes.length > 0, '管理模式每行应当有选择框')
    assert.equal(
      view.queryAll('.conversation-list input').length,
      checkboxes.length,
      '编辑框应当收起来，只剩下每行的选择框'
    )
  } finally {
    await close(view)
  }
})

test('重命名 R11：编辑态内点击不切换会话', async () => {
  const { view, router } = await mountLayout({ route: '/chat' })
  const { useChatStore } = await import(
    new URL('../src/stores/chat.js', import.meta.url).href
  )

  try {
    const chatStore = useChatStore(view.pinia)
    const beforeSelect = chatStore.currentId

    const input = await openRename(view, 0)
    input.click()
    await settle(view)

    assert.equal(
      router.currentRoute.value.path,
      '/chat',
      '编辑态内点击不应当把路由推到会话详情'
    )
    assert.equal(chatStore.currentId, beforeSelect, '编辑态内点击不应当改写当前会话')
    assert.ok(view.query('.conversation-list input'), '编辑态内点击不应当把编辑框点没了')
  } finally {
    await close(view)
  }
})

test('重命名 R12：改名之后当即行序不变（重排发生在下次拉取列表）', async () => {
  const { view } = await mountLayout()

  try {
    const before = conversationRows(view).map((row) => row.textContent)
    assert.match(before[0], /第一段对话/, '改名之前第一行应当是第一条会话')

    const input = await openRename(view, 0)
    typeInto(input, '改名后的第一段')
    pressKey(input, 'Enter')
    await settle(view)

    const after = conversationRows(view).map((row) => row.textContent)
    assert.equal(after.length, before.length, '行数不应当变化')
    assert.match(after[0], /改名后的第一段/, '改名的那一行应当还在原位')
    assert.match(after[1], /第二段对话/, '另一行的位置与内容都不应当被牵动')
    // 自足：这条用例不与 R3 共用一次 mount，否则「行序没变」可能只是巧合地成立。
    assert.equal(putCalls().length, 1, '这条用例自己发过一次提交')
  } finally {
    await close(view)
  }
})

test('点击历史对话只做导航，消息加载交给 Chat.vue 的路由 watcher', async () => {
  const { view, router } = await mountLayout({ route: '/chat' })
  const { useChatStore } = await import(
    new URL('../src/stores/chat.js', import.meta.url).href
  )

  try {
    const chatStore = useChatStore(view.pinia)
    assert.equal(chatStore.currentId, null, '前置：还没有任何会话被加载')

    conversationRows(view)[0].click()
    await until(view, () => router.currentRoute.value.path === '/chat/c1')
    await settle(view)

    assert.equal(router.currentRoute.value.path, '/chat/c1', '点击会话行应当导航到该会话')
    // 侧栏不再自己调 selectConversation：加载失败时要由路由层统一回滚，
    // 两处各发一次请求会让回滚拿到两份互相追逐的结果。
    assert.equal(chatStore.currentId, null, '侧栏不应当自行选中/加载会话')
    assert.equal(
      calls.filter((entry) => entry.method === 'get' && entry.args[0] === '/chat/conversations/c1').length,
      0,
      '侧栏点击不应当自己发消息请求'
    )
  } finally {
    await close(view)
  }
})

test('重命名 R13：IME 组字中的回车不提交', async () => {
  const { view } = await mountLayout()

  try {
    const input = await openRename(view, 0)
    typeInto(input, '半成品输入')
    pressKey(input, 'Enter', { isComposing: true })
    await settle(view)

    assert.equal(putCalls().length, 0, '组字中的回车不应当提交')
    assert.equal(view.messages.length, 0, '组字中的回车不应当弹提示')
    const stillEditing = view.query('.conversation-list input')
    assert.ok(stillEditing, '组字中的回车之后应当仍处于编辑态')
    // 编辑态里标题在输入框的 value 上、不在 textContent 里，所以这里钉 value。
    assert.equal(stillEditing.value, '半成品输入', '草稿应当原样留在输入框里')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// e. 标题检索（issue #251）：输入防抖到一次请求，空结果有专门的文案
// ---------------------------------------------------------------------------

test('标题检索 S1：输入防抖后只发一次带 q 的请求，空结果提示「未找到匹配的对话」', async () => {
  const { view } = await mountLayout()

  try {
    const search = view.query('.conversation-search input')
    assert.ok(search, '「历史对话」下方应当渲染标题检索输入框')
    assert.equal(search.getAttribute('placeholder'), '搜索对话标题')

    // 带 q 的请求返回空列表；默认列表请求仍返回挂载时的两条。
    respond('get', (url, config) => {
      if (url === '/chat/conversations') {
        return config?.params?.q ? [] : [CONV1, CONV2]
      }
      if (url === '/user/profile') return PROFILE
      return {}
    })

    const before = conversationListCalls().length

    // 三次输入都落在同一个 300ms 防抖窗口里，最终应当只落成一次请求。
    typeInto(search, '知')
    typeInto(search, '知识')
    typeInto(search, '知识库')
    assert.equal(conversationListCalls().length, before, '防抖窗口内不应当发请求')

    // 真计时器（不用 fake timers）：等防抖到期，但只等请求本身，不 sleep 固定时长。
    const sent = await until(view, () => conversationListCalls().length > before, 2000)
    assert.ok(sent, '停止输入后应当发出检索请求')

    const searched = conversationListCalls().slice(before)
    assert.equal(searched.length, 1, '连续输入只应当落成一次请求')
    assert.deepEqual(searched[0].args[1].params, { limit: 51, q: '知识库' })

    const rendered = await until(view, () => view.text().includes('未找到匹配的对话'), 2000)
    assert.ok(rendered, '检索无结果时应当显示「未找到匹配的对话」')
    assert.doesNotMatch(view.text(), /第一段对话/, '旧列表应当被检索结果替换')
    await settle(view)
    assert.equal(conversationListCalls().length, before + 1, '检索本身不应触发额外请求')
  } finally {
    await close(view)
  }
})
