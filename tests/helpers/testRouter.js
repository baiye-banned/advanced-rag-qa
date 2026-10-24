// 挂载用例的最小路由。
//
// Knowledge.vue 现在会读 `route.query.file` 来恢复详情预览（issue #255），即使某个用例
// 并不关心地址，也必须让 useRoute()/useRouter() 拿到真的路由对象 —— 没装 vue-router 时
// 它们返回 undefined，症状会漂成「读 route.query 时抛」，看不出是路由没装。
//
// 路由表用通配：这些用例验证的是创建/重命名/删除/上传的调用点，不应该顺带依赖具体路径。
// 需要从某个地址起步、并断言地址变化的用例（tests/knowledgeViewMount.test.js）把初始
// 路径传进来即可，路由表本身不必知道具体页面。
export async function createTestRouter(initialPath = '/knowledge') {
  const { createMemoryHistory, createRouter } = await import('vue-router')
  const router = createRouter({
    history: createMemoryHistory(),
    routes: [{ path: '/:pathMatch(.*)*', component: { render: () => null } }],
  })
  await router.push(initialPath)
  await router.isReady()
  return router
}
