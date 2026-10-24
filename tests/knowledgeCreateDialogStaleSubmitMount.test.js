// Knowledge.vue 创建对话框跨请求时序的挂载用例：真 mount 视图、点真按钮、填真输入框。
//
// 覆盖的缺陷（issue #156）：点「创建」之后又点「取消」，创建请求并不会跟着取消，它仍在飞；
// 用户随即重新打开对话框并开始输入下一个名字时，上一笔请求的迟到响应会走完成功分支，
// 无条件把 `knowledgeBaseDialogVisible` 置回 false —— 用户刚打开的对话框被关掉，
// 已经输入的名称一并丢失。
//
// 与既有用例的分工：
//   tests/detailPreview.test.js、tests/fileListRequest.test.js
//                                    纯函数级，钉的是两个 util 里同款序号守卫的形态
//   tests/knowledgeViewWiring.test.js 静态读文件，证明「视图里写了这几行」
//   本文件                             视图真的跑起来之后，对话框的开关由谁决定
//
// 替身只扎在模块边界 `api/request.js` 上（同 knowledgeDeleteCallSiteMount.test.js）：
// knowledgeAPI、pinia store、fileListRequest 与视图本身全是真货，所以「对话框有没有被关掉」
// 是组件真实响应式状态驱动的 DOM 结果，不是断言一个被 stub 的开关。
//
// 观测口径：模板 :146 的 el-dialog 带 destroy-on-close，关闭动画结束后内容会被销毁，
// 于是 `.knowledge-base-form-dialog` 还在 host 里 = 对话框仍处于打开态。

import test from 'node:test'
import assert from 'node:assert/strict'
import { mountSfc } from './helpers/vueMount.js'
import { createTestRouter } from './helpers/testRouter.js'
import { callsOf, resetRequestStub, respond } from './helpers/stubApiRequest.js'

const MODULES = {
  'api/request.js': new URL('./helpers/stubApiRequest.js', import.meta.url).href,
}

const KB1 = { id: 'kb1', name: 'KB1' }
const ONE_FILE = [{ id: 'f1', name: 'a.pdf', size: 10, created_at: '2024-01-01T00:00:00Z' }]

const dialogOpen = (view) => !!view.query('.knowledge-base-form-dialog')

// 对话框的开关要跨过 el-dialog 的过渡：flush 的每一轮只是 microtask + setTimeout(0)，
// 而 jsdom 的 requestAnimationFrame 约 16ms 才触发，`@closed`（reset 挂在这里）就在那条路上。
// 这个缺口的表现是「机器越快越容易红」，所以除了轮数还要给一点真实时间。
const settle = async (view) => {
  await view.flush(10)
  await new Promise((resolve) => setTimeout(resolve, 40))
}

// 等一个可观测条件成立，超时就把当前状态原样交给后面的断言去报。
async function until(view, predicate, budgetMs = 1000) {
  const deadline = Date.now() + budgetMs
  for (;;) {
    if (predicate()) return true
    if (Date.now() > deadline) return false
    await view.flush(1)
    await new Promise((resolve) => setTimeout(resolve, 2))
  }
}

// 一次挂载 = 一次完整的视图启动（onMounted 会拉知识库列表与文件列表）。
async function mountKnowledge() {
  resetRequestStub()
  respond('get', (url) => {
    if (url === '/knowledge-bases') return [KB1]
    if (url === '/knowledge') return ONE_FILE
    return {}
  })

  const router = await createTestRouter()
  const view = await mountSfc('views/Knowledge.vue', { modules: MODULES, plugins: [router] })
  await settle(view)
  // 挂载期有两次 GET。等它们都落地，否则迟到的挂载请求会混进后面的观测里。
  await until(view, () => callsOf('get').length >= 2)
  return view
}

async function openCreateDialog(view) {
  assert.ok(view.buttonByText('新建知识库'), '页面上应当有「新建知识库」按钮')
  view.buttonByText('新建知识库').click()
  await until(view, () => dialogOpen(view))
  assert.ok(dialogOpen(view), '点「新建知识库」后对话框应当打开')
}

// 往名称输入框里打字。返回输入框本身，便于后面直接读回它的 value 判断内容有没有被丢掉。
function typeName(view, name) {
  const input = view.query('.knowledge-base-form-dialog input.el-input__inner')
  assert.ok(input, '创建对话框里应当渲染出名称输入框')
  input.value = name
  input.dispatchEvent(new Event('input', { bubbles: true }))
  return input
}

// 点对话框底部的「创建」。断言按钮确实在，否则「没有发出请求」会伪装成别的原因。
async function submitCreate(view) {
  const button = view.buttonByText('创建')
  assert.ok(button, '对话框底部应当有提交按钮')
  button.click()
  await view.flush(3)
}

async function cancelDialog(view) {
  const button = view.buttonByText('取消')
  assert.ok(button, '对话框底部应当有「取消」按钮')
  button.click()
  await until(view, () => !dialogOpen(view))
  assert.equal(dialogOpen(view), false, '点「取消」后对话框应当关闭')
}

async function close(view) {
  // ElMessage 的补丁挂在 element-plus 的模块对象上，跨用例共享：
  // 断言中途失败也必须还原，否则残留的补丁会污染后续用例记到的提示。
  view.elementPlusUnpatch()
  await view.unmount()
}

// ---------------------------------------------------------------------------
// a. 迟到响应：取消后重开，上一笔提交的响应不得再动这个对话框（承重：序号守卫）
// ---------------------------------------------------------------------------

test('取消后重新打开：上一笔在飞创建的迟到响应不得关掉新对话框，也不得丢掉已输入的名称', async () => {
  const view = await mountKnowledge()
  try {
    // 手动持有兑现权：让第一笔创建停在在飞态，由用例决定它什么时候返回。
    let deliver
    respond('post', () => new Promise((resolve) => { deliver = resolve }))

    await openCreateDialog(view)
    typeName(view, '新库')
    await view.flush(3)
    await submitCreate(view)

    // 控制项①：创建请求真的发出去了并且停在在飞态。少了它，后面全是空转的绿灯。
    await until(view, () => callsOf('post').length === 1)
    assert.equal(callsOf('post').length, 1, '点「创建」应当恰好发出一次创建请求（控制项）')
    assert.equal(typeof deliver, 'function', '创建请求应当挂在用例手里的 pending Promise 上（控制项）')

    // 取消。请求不会跟着取消 —— 这正是本用例的前提。
    await cancelDialog(view)

    // 重新打开，并开始输入第二个知识库的名字（模拟用户已经在继续操作）。
    await openCreateDialog(view)
    const reopenedInput = typeName(view, '第二个库')
    await view.flush(3)

    // 上一笔创建此刻才返回。它会成功，且带着一个真实的 created 对象。
    deliver({ id: 'kb9', name: '新库' })
    await settle(view)

    // 承重断言：新对话框不该被上一笔提交的响应关掉，用户输入的内容也不该被清掉。
    assert.deepEqual(
      {
        dialogClosedByLateResponse: !dialogOpen(view),
        typedTextStillThere: reopenedInput.value,
      },
      { dialogClosedByLateResponse: false, typedTextStillThere: '第二个库' },
      '【#156】用户刚重新打开、并已输入名称的创建对话框，被上一笔在飞提交的迟到响应关掉了'
    )

    // 控制项②：重开的这个对话框本身是活的 —— 它自己发起的提交照常成功、照常关窗。
    // 少了它，上面那条「什么都没发生」可能只是因为对话框已经坏掉，而不是守卫生效。
    respond('post', () => ({ id: 'kb10', name: '第二个库' }))
    await submitCreate(view)
    await settle(view)
    assert.equal(dialogOpen(view), false, '新会话自己发起的创建应当照常关窗（控制项）')
    assert.ok(
      view.messages.some((m) => m.level === 'success' && m.message === '知识库已创建'),
      '新会话自己发起的创建应当照常弹成功提示（控制项）'
    )
  } finally {
    await close(view)
  }
})

test('迟到响应不作数：它在对话框关闭期间返回时，同样不该改写当前知识库与提示', async () => {
  const view = await mountKnowledge()
  try {
    let deliver
    respond('post', () => new Promise((resolve) => { deliver = resolve }))

    await openCreateDialog(view)
    typeName(view, '新库')
    await view.flush(3)
    await submitCreate(view)
    await until(view, () => callsOf('post').length === 1)

    await cancelDialog(view)
    // 不重开：这一笔提交所属的对话框会话已经结束，它的响应不该再驱动任何界面状态。
    deliver({ id: 'kb9', name: '新库' })
    await settle(view)

    assert.deepEqual(
      {
        dialogOpen: dialogOpen(view),
        messages: view.messages,
        currentStillKb1: view.text().includes('KB1（0）'),
      },
      { dialogOpen: false, messages: [], currentStillKb1: true },
      '【#156】已取消的那笔提交，其迟到响应不该再改写对话框、提示与当前知识库'
    )
  } finally {
    await close(view)
  }
})

test('迟到响应不得复位新会话的提交中状态：新会话自己那一笔还在飞', async () => {
  const view = await mountKnowledge()
  try {
    // 两笔提交各自的兑现权，按发起顺序入队。
    const pending = []
    respond('post', () => new Promise((resolve) => { pending.push(resolve) }))

    await openCreateDialog(view)
    typeName(view, '第一个库')
    await view.flush(3)
    await submitCreate(view)
    await until(view, () => pending.length === 1)

    await cancelDialog(view)

    // 重开后立刻发起第二笔：此时它才是「当前」的那一笔。
    await openCreateDialog(view)
    typeName(view, '第二个库')
    await view.flush(3)
    await submitCreate(view)
    await until(view, () => pending.length === 2)

    const createButton = view.buttonByText('创建')
    assert.ok(createButton, '对话框底部应当有提交按钮')
    assert.equal(
      createButton.classList.contains('is-loading'),
      true,
      '第二笔提交在飞时「创建」按钮应当处于提交中态（控制项）'
    )

    // 第一笔（已经作废的那一笔）此刻才返回。
    pending[0]({ id: 'kb9', name: '第一个库' })
    await settle(view)

    // 承重断言：它的 finally 不能把第二笔的提交中态复位掉。
    assert.deepEqual(
      {
        dialogOpen: dialogOpen(view),
        stillSubmitting: view.buttonByText('创建')?.classList.contains('is-loading'),
      },
      { dialogOpen: true, stillSubmitting: true },
      '【#156】上一笔作废提交的迟到响应复位了新会话正在进行的提交状态'
    )

    // 收尾：第二笔自己返回时照常关窗。
    pending[1]({ id: 'kb10', name: '第二个库' })
    await settle(view)
    assert.equal(dialogOpen(view), false, '第二笔自己返回后应当照常关窗')
  } finally {
    await close(view)
  }
})

test('关闭过渡还没走完就重新打开：作废不能只挂在关闭钩子上', async () => {
  const view = await mountKnowledge()
  try {
    let deliver
    respond('post', () => new Promise((resolve) => { deliver = resolve }))

    await openCreateDialog(view)
    typeName(view, '新库')
    await view.flush(3)
    await submitCreate(view)
    await until(view, () => callsOf('post').length === 1)

    // 取消与重开挨在一起：el-dialog 的 @closed（reset 挂在这里）要等关闭过渡真正跑完，
    // 而紧接着的重开会把这次关闭整个抵消掉，于是这条路上一次自增都没有发生过。
    // 打开对话框本身就必须作废在飞提交，否则守卫的形状就被绑死在关闭钩子上。
    view.buttonByText('取消').click()
    view.buttonByText('新建知识库').click()
    await until(view, () => dialogOpen(view))
    assert.ok(dialogOpen(view), '重开后对话框应当处于打开态（控制项）')
    const input = typeName(view, '第二个库')
    await view.flush(3)

    deliver({ id: 'kb9', name: '新库' })
    await settle(view)

    assert.deepEqual(
      { dialogOpen: dialogOpen(view), typedText: input.value },
      { dialogOpen: true, typedText: '第二个库' },
      '【#156】关闭过渡未结束就重开时，上一笔提交的迟到响应仍然关掉了新对话框'
    )

    // 控制项：新会话不能只是「还开着」，它得真的能用。作废那一笔的 finally 不再复位
    // 提交中态，而唯一复位它的 reset 挂在 `@closed` 上 —— 偏偏这条路上 `@closed` 不触发，
    // 于是「作废」若只顾着把响应丢掉、不管这份状态，对话框就会停在 loading：
    // 按钮转着圈，提交又被 knowledgeBaseSubmitting 守卫挡住，开着却用不了。
    const createButton = view.buttonByText('创建')
    assert.ok(createButton, '对话框底部应当有提交按钮')
    assert.equal(
      createButton.classList.contains('is-loading'),
      false,
      '新会话不该继承上一笔作废提交的提交中态（控制项）'
    )

    respond('post', () => ({ id: 'kb10', name: '第二个库' }))
    await submitCreate(view)
    await settle(view)
    assert.equal(dialogOpen(view), false, '新会话自己发起的创建应当照常关窗（控制项）')
  } finally {
    await close(view)
  }
})

// ---------------------------------------------------------------------------
// b. 对照：响应在取消之前就到达时，一切都照旧（守卫不得误杀合法提交）
// ---------------------------------------------------------------------------

test('对照：响应在取消前到达时照常关窗并提示，重开的对话框不受影响', async () => {
  const view = await mountKnowledge()
  try {
    respond('post', () => ({ id: 'kb9', name: '新库' }))

    await openCreateDialog(view)
    typeName(view, '新库')
    await view.flush(3)
    await submitCreate(view)
    await settle(view)

    // 这一笔提交在取消之前就收尾了：它正常关窗、正常提示 —— 与缺陷场景的唯一差别是到达时机。
    assert.equal(dialogOpen(view), false, '创建成功后对话框应当关闭')
    assert.deepEqual(
      view.messages,
      [{ level: 'success', message: '知识库已创建' }],
      '创建成功后应当恰好弹一次成功提示'
    )

    // 此时已无在飞提交，重开的对话框必须保持打开。
    await openCreateDialog(view)
    await settle(view)
    assert.ok(dialogOpen(view), '没有在飞响应时，重开的对话框应当保持打开（对照）')
  } finally {
    await close(view)
  }
})

test('连续创建两次：两笔各自关掉自己的对话框，守卫不误杀合法的连续提交', async () => {
  const view = await mountKnowledge()
  try {
    respond('post', () => ({ id: 'kb9', name: '第一个新库' }))
    await openCreateDialog(view)
    typeName(view, '第一个新库')
    await view.flush(3)
    await submitCreate(view)
    await settle(view)
    assert.equal(dialogOpen(view), false, '第一次创建成功后应当关窗')

    respond('post', () => ({ id: 'kb10', name: '第二个新库' }))
    await openCreateDialog(view)
    typeName(view, '第二个新库')
    await view.flush(3)
    await submitCreate(view)
    await settle(view)
    assert.equal(dialogOpen(view), false, '第二次创建成功后同样应当关窗')
    assert.deepEqual(
      view.messages.filter((m) => m.level === 'success').map((m) => m.message),
      ['知识库已创建', '知识库已创建'],
      '两次创建各弹一次成功提示'
    )
  } finally {
    await close(view)
  }
})
