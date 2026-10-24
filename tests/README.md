# Test Baseline

This directory contains the executable regression baseline for the current RAG maintenance phase.

**When you add a test file, register it in the two lists below in the same PR.** Both lists are
hand-maintained: `npm test` discovers files through `tests/**/*.test.js` and pytest through
`python_files = test_*.py` in `pytest.ini`, so a missing entry never stops a test from running - it
only makes that file invisible to anyone auditing coverage against this document, which is how an
already-covered behaviour ends up being read as uncovered. The directory itself is the source of
truth; when a list and the tree disagree, `ls tests/*.test.js tests/test_*.py tests/conftest.py`
and `python -m pytest tests --collect-only` win. (`tests/e2e/*.spec.mjs` is the Playwright suite
run by `npm run test:e2e` and is deliberately in neither list.)

The two `These tests cover ...` notes are soft-wrapped on purpose: they are folded at `100` columns
rather than one clause per line, so a registration adds a new line at the end of a note instead of
rewriting one 15k-character line - which is what made any two concurrent registrations collide on
the same line and lose one side's entry to a "take one side" conflict resolution (issue #202). Add
your clause on a new line and then run `node scripts/tests_readme_magnet.mjs format`: the notes
must sit in exactly the form that command produces, and `npm test` fails when they do not - so a
hand edit that skips `format` turns CI red even when every line is under the cap. That cap is all
`node scripts/tests_readme_magnet.mjs check` looks at: it is a quick editing-time helper for the
gross regression (a note collapsing back into one line), not the gate. The gate is the
`testsReadmeMagnet.test.js` ratchet, which runs under `npm test` in CI. If a merge does conflict,
`node scripts/tests_readme_magnet.mjs union <file>` merges both sides in one command; it refuses to
write - leaving the file untouched - whenever it cannot prove that neither side lost a clause, or
that both sides did not rewrite the same spot of one line (a union there would print that sentence
twice). A shape it can only *suspect* is written, but the command says so on stderr - and "suspect"
has a definition: the two sides' conflicting lines share at least 24 characters. Below that the
union is silent, deliberately, and one case is worth knowing about because it is silent by design
rather than by oversight: when both conflicting lines are shorter than 24 characters the shared run
cannot reach the threshold, so a same-spot rewrite there gets neither a refusal nor a warning (the
text that gets written twice is under 24 characters, and on lines that short the two readings are
not separable - measured, a same-spot pair and an ordinary pair of appends both land between 0.5
and 0.95 shared-run ratio, so any threshold would warn on ordinary unions).

## Node Tests

Requires Node.js **>= 22.15**. `chatStore.test.js` imports `module.registerHooks`, which Node
documents as *Added in: v23.5.0, v22.15.0*; on Node 18/20 and 22.0-22.14 that import fails and
`node --test` counts the file as failed, so the whole `npm test` run exits non-zero - not just
that one file. The 23.x line needs **23.5+** (23.0.0-23.4.x lacks the export); 24+ is fine.

The floor is declared in `engines.node` (`>=22.15`) in `package.json` and in the environment
list in `README.md`; keep the three in sync. `npm install` / `npm ci` prints an `EBADENGINE`
warning when the running Node does not satisfy `engines.node`. Because `>=22.15` is a plain
floor it also admits the EOL 23.0.0-23.4.x releases, which this check rejects:

```bash
node -e "const [M,m]=process.versions.node.split('.').map(Number); process.exit((M===22&&m>=15)||(M===23&&m>=5)||M>=24?0:1)"
```

The workflows in `.github/workflows/` pin `node-version: "22"`, which resolves to the latest
22.x and therefore sits above the floor - nothing in CI runs at 22.15.0 itself. To cover the
floor locally, pin it with any version manager and run the suite from the repository root:

```bash
nvm install 22.15.0 && nvm use 22.15.0    # or fnm/asdf with the same version
npm ci && npm test                        # exits 0
nvm use 22.14.0 && npm test               # exits 1: chatStore.test.js fails to load
```

Re-run this whenever `engines.node` or the tests' API usage changes; 22.14.0 is the cheap
half of the check, since a floor that is too low only shows up on the version below it.

Run from the repository root:

```powershell
npm test
```

The test script uses `tests/**/*.test.js`, so future nested Node test files that
end in `.test.js` are included as well.

Current files:

- `chatApi.test.js`
- `chatExport.test.js`
- `chatExportRequest.test.js`
- `chatStore.test.js`
- `chatStorePaging.test.js`
- `chatViewMount.test.js`
- `ciGateScripts.test.js`
- `clipboard.test.js`
- `detailPreview.test.js`
- `fileListRequest.test.js`
- `knowledgeCreateCallSiteMount.test.js`
- `knowledgeCreateDialogStaleSubmitMount.test.js`
- `knowledgeDeleteCallSiteMount.test.js`
- `knowledgeFeedback.test.js`
- `knowledgeRenameCallSiteMount.test.js`
- `knowledgeStore.test.js`
- `knowledgeUploadCallSiteMount.test.js`
- `knowledgeUploadTypes.test.js`
- `knowledgeViewMount.test.js`
- `knowledgeViewWiring.test.js`
- `layoutViewMount.test.js`
- `loginViewMount.test.js`
- `markdownRendererMount.test.js`
- `markdownSanitize.test.js`
- `testsReadmeMagnet.test.js`
- `streamEvents.test.js`
- `traceVariableFlowMount.test.js`
- `url.test.js`
- `userAvatarSrc.test.js`
- `userProfileViewMount.test.js`
- `utils.test.js`

These tests cover frontend stream parsing, stream response errors, chat store conversation
switching, stale in-flight response handling and fallback reset events, knowledge file detail
preview ordering (a late response from a previously opened file must not rewrite the current one,
and closing the dialog discards in-flight requests), clipboard fallback behavior, knowledge batch
delete and upload feedback helpers, the shared delete confirmation orchestration (user cancel, API
rejection, and success outcomes), the upload-type whitelist staying in sync with the backend's
accepted extensions (a static cross-check against `backend/service/utils_service.py`), HTML
sanitization of rendered markdown (script/style/event-handler stripping, the element allowlist, and
disallowed elements being dropped together with their content), URL normalization and link safety,
display formatting, status helpers, image validation, the CI gate scripts
(`scripts/check_issue.mjs`, `scripts/check_pr_body.mjs` and their shared
`scripts/lib/markdown_sanitize.mjs`: the comment sanitizer's invariants - no comment start marker
left behind, no section heading deleted by an unpaired marker - plus both gates' pass/fail verdicts
on the fixtures in `tests/fixtures/ci-gate/`, including the code-fence behaviour that is deliberate
rather than an oversight), memory trace helpers, and the mounted-SFC layer for
`TraceVariableFlow.vue` (`traceVariableFlowMount.test.js`: `@vue/compiler-sfc` compiles the real
SFC, jsdom hosts it and Element Plus supplies the real `el-switch`, so the assertions cover the
main-line/branch classification, the only-main-line and expand-branch switch linkage including the
reset of the expand state, the variable inspector's selection, same-name upstream/downstream wiring
and full-value rendering, and the clipboard toast paths - see the file header for which of the
component's two state-sync watchers is reachable from the UI and which one is not), and the delete
call-site glue that only exists once the view is real (`knowledgeDeleteCallSiteMount.test.js`: the
actual `Knowledge.vue` is mounted and its actual buttons are clicked, with only one module boundary
replaced - the HTTP egress in `api/request.js` - so the assertions land on which requests were
really issued and which toasts were really raised, namely that the `status !== DELETE_SUCCEEDED`
short-circuit really does block the follow-up refresh and that the refresh really is sent when it
does not, across all three delete entry points; it is the layer between `knowledgeFeedback.test.js`,
which calls the helpers directly with injected doubles, and `knowledgeViewWiring.test.js`, which
only reads the source statically), and the create dialog's cross-request ordering
(`knowledgeCreateDialogStaleSubmitMount.test.js`: the real `Knowledge.vue` is mounted with only the
HTTP egress replaced, so whether the dialog is open is read off the component's own reactive state
rather than a stubbed flag; issue #156 is that cancelling a create does not cancel the request, so
when the user reopened the dialog and started typing a new name the late response still ran the
success branch and closed it unconditionally - the cases pin a monotonic submit sequence that
invalidates the in-flight submission both when the dialog is opened and when it is closed, and the
controls prove the guard is not bought by breaking anything else, namely that the same late response
is still applied when the request finished before the cancellation, that back-to-back creates each
close their own dialog, that a late response must not clear a newer submission's in-flight state,
and that a guard which invalidated every submission would be caught as killing legitimate ones; the
reopening case additionally pins that invalidating the submission leaves the new session usable
rather than inheriting its in-flight state - the abandoned submission's `finally` no longer clears
the submitting flag and the reset that does is on the dialog's close hook, which is precisely the
hook that path skips). The mounted-SFC layer also covers `Knowledge.vue`'s upload entry
(`knowledgeUploadCallSiteMount.test.js`: a real `change` event on the hidden file input drives the
whole chain, and the assertions pin the complete `ElMessage` array rather than a subset, so an
upload whose follow-up list refresh fails keeps the success toast and adds only a refresh-failure
message, while a genuine upload failure still reports as one). `userAvatarSrc.test.js` is the
browser-side half of issue #186 (`test_uploads_anonymous_read_186.py` proves the route still serves
the owner who brings a token, but `<img src>` cannot carry an `Authorization` header, so the same
fix would also hide every user's own avatar - a regression no backend case can see), and it runs the
real store with only `api/user.js` and `api/auth.js` replaced: the avatar is fetched with the token
and handed to `<img>` as an object URL, a missing avatar or one that is not a local path issues no
request at all, a failed fetch does not bubble and leaves the initial-letter fallback, replacing an
avatar revokes the old object URL while a superseded fetch's late response never produces one,
`logout()` revokes and clears, and both avatar-rendering views (`Layout.vue` and `UserProfile.vue`)
are read statically to pin that they bind the store's `avatarSrc` rather than rebuilding the stored
path - the layer where a view that still used the stored path would keep every assertion above green
while the avatar stayed 401 in the browser, and the rest of the SFC mount layer from issue #180,
whose gap was that `git grep mountSfc tests/` reached only two components (`Knowledge.vue` and
`TraceVariableFlow.vue`): `knowledgeRenameCallSiteMount.test.js` drives the rename branch end to end -
the current base is switched to the second one first so that "the selection stayed put" cannot
pass by coincidence - and pins the name prefill, the same-name short-circuit, that the sidebar and
the current selection both follow the renamed base, and that a failing sidebar refetch still upserts
the renamed base without leaking an unhandled rejection; `markdownRendererMount.test.js` mounts the
renderer itself, so "the HTML it produces is sanitised" is proven by execution instead of by the
comment in `markdownSanitize.test.js` (an observing wrapper around `utils/sanitizeHtml.js` records
what the renderer handed in and what came back, the mounted DOM must equal the sanitised output, and
a control case with the sanitiser bypassed proves the payload really would reach the DOM, so the
negative assertions cannot pass vacuously); `chatViewMount.test.js` drives a question through a real
streaming response and reads the answer off the real `MarkdownRenderer` child;
`layoutViewMount.test.js` clicks the sidebar menu and pins that the route and the active item move
together, and drives the sidebar's inline rename (issue #257) end to end - the hover-revealed entry,
the prefill and the autofocus, Enter sending exactly one `PUT` and the new title showing in place
while the row order deliberately stays put until the next list fetch, Esc and blur cancelling
without a request, the whitespace and same-name short-circuits, a failed `PUT` keeping the old title
and toasting an error, the composing-Enter guard, and the in-editor click not switching
conversations; `loginViewMount.test.js` covers the credential payload, the form-level error, the
`redirect` target and that real validation blocks the request; `userProfileViewMount.test.js` covers
the password change both ways (success clears all three fields and sends neither the confirmation
field nor anything else, failure keeps them) plus the avatar's local file validation; and
`knowledgeDeleteCallSiteMount.test.js` no longer replaces `utils/confirm.js` at all, so both of the
confirm dialog's exits - cancel returning silently and confirm driving the delete plus its refresh -
are executed against the real `ElMessageBox`, whose stated blocker in `helpers/stubConfirm.js` (a
missing `HTMLInputElement` global) had already been removed by the mount helper's element-class list
and was therefore stale rather than true. `fileListRequest.test.js` also carries the knowledge file
list's on-demand paging, which is the half of issue #191 that keeps the backend's new page-size cap
from making older files disappear without a trace: the first request asks for one row more than a
page so `exactly full` and `already exhausted` stay distinguishable, the cursor is the last id of
the loaded range rather than of the array, a page that arrives stale after a knowledge-base switch
is dropped instead of being appended to the new base's list, and once a short page has said there is
nothing more, a further `loadMore` issues no request at all. `knowledgeStore.test.js` imports the
knowledge store directly and closes the store-side half of the same issue: a second
`fetchKnowledgeBases()` is served from the cache without touching the network and, because the
short-circuit sits before the loading flag, must not flip `loading` either,
`force`/`refreshKnowledgeBases` bypass the cache and replace the list wholesale,
`upsertKnowledgeBase` merges in place on a known id (length and order unchanged) and appends on a
new one, rows that normalise to null are filtered out of the list instead of being rendered - an
object with no id at all counts as one of those, so a payload that is missing its id neither appends
a nameless empty row nor flips the empty-state flag - and the knowledge-base paging entry
(`loadMoreKnowledgeBases`) takes its cursor from the page it just fetched rather than from the end
of the array - which is what keeps it advancing instead of spinning in place when a whole page turns
out to be already loaded - appends only ids it does not already hold, drops the `hasMore` flag as
soon as a short page says there is nothing left, refuses a second request while one is in flight,
and offers no "load more" entry at all when a page normalises down to no rows and no last id is left
to build a cursor from - that cursor is read off the normalised page, so a row that is missing its
id no longer strands the cursor: the append drops it and the cursor still lands on the last real id,
which is what keeps one malformed row from silently truncating the list, on any page.
`chatStorePaging.test.js` is the sidebar half of issue #191, whose review asked for exactly this
coverage: the first request asks for one row more than a page so `exactly full` and `already
exhausted` stay distinguishable, the cursor it hands back is the `{id, updated_at}` pair of the
page's last row (a conversation id alone cannot order a uuid-keyed list), overlapping rows that a
refresh or a stream tail inserted are dropped id by id rather than duplicated, a page whose last row
carries no `updated_at` produces no entry at all instead of a button that does nothing, and
re-fetching replaces the list with the newest page and resets the cursor rather than continuing from
the stale one. Conversation Markdown export (#252) preserves the full Axios response only for the
download request, parses UTF-8 filenames, rejects non-Markdown payloads, releases object URLs, and
keeps ordinary JSON requests and login expiry handling unchanged. Mounted Chat.vue tests cover the
real export button, unavailable states, duplicate clicks, errors and conversation changes while a
download is pending. `knowledgeViewMount.test.js` mounts the real `Knowledge.vue` behind a real
memory router and pins the address contract added for issue #255: `/knowledge?file=<positive id>`
opens that file's detail both when the file is already in the list (without refetching its metadata)
and when it has to be fetched first, a metadata 404 warns that the file does not exist or was
deleted and clears only the `file` query while keeping the other parameters, a network or
content-read failure keeps the query and is never misreported as a missing file, a late metadata
response for a superseded id cannot open the wrong file, and clicking a row's own detail button
still writes no `file` query.

## Python Tests

Run from the repository root:

```powershell
python -m pytest -q tests
```

Discovery is fixed by the repository `pytest.ini`: `testpaths = tests` and
`python_files = test_*.py`.

Current files:

- `test_attachment_lifecycle.py`
- `test_auth_service.py`
- `test_bm25_288.py`
- `test_boundary_modules.py`
- `test_chat_export_252.py`
- `test_chat_service_retrieval.py`
- `test_checkpointer.py`
- `test_checkpointer_threads_removed_236.py`
- `test_chunk_key_namespace.py`
- `test_chunking.py`
- `test_config_helpers.py`
- `test_conversation_activity_bump_239.py`
- `test_conversation_title_whitespace_265.py`
- `test_cors_credentials_235.py`
- `test_default_users.py`
- `test_empty_extraction_166.py`
- `test_empty_model_response_259.py`
- `test_env_example_parity.py`
- `test_eval_harness_293.py`
- `test_grounding.py`
- `test_image_upload_octet_stream_237.py`
- `test_internal_error_no_echo.py`
- `test_json_utils.py`
- `test_keyword_recall_memory_59.py`
- `test_knowledge_base_conflict_diagnosis.py`
- `test_knowledge_base_name_race.py`
- `test_knowledge_delete_load_only_179.py`
- `test_knowledge_files_index.py`
- `test_knowledge_ownership.py`
- `test_knowledge_service.py`
- `test_knowledge_upload_limits.py`
- `test_learning_trace.py`
- `test_list_page_caps_191.py`
- `test_list_query_counts.py`
- `test_llm_urls.py`
- `test_memory_context.py`
- `test_message_conversation_indexes_176.py`
- `test_message_feedback_250.py`
- `test_milvus_acceptance.py`
- `test_milvus_client.py`
- `test_mysql_password_guard.py`
- `test_orphan_attachment_cleanup_142.py`
- `test_pdf_extraction.py`
- `test_provider_smoke.py`
- `test_ragas_eval.py`
- `test_rate_limit_183.py`
- `test_rerank.py`
- `test_retrieval.py`
- `test_retrieval_acceptance.py`
- `test_scan_secrets_selftest.py`
- `test_secret_key_guard.py`
- `test_sse_session_leak.py`
- `test_stream_fallback_reset.py`
- `test_sync_session_off_loop_187.py`
- `test_token_revocation_184.py`
- `test_trace_crud.py`
- `test_trace_index_after_conversation_delete_141.py`
- `test_trace_purge_on_conversation_delete_127.py`
- `test_trace_writeback_off_loop_201.py`
- `test_upload_validation.py`
- `test_uploads_anonymous_read_186.py`
- `test_vision_outbound_guard_181.py`
- `conftest.py`

These tests cover retrieval planning and fusion, chat service retrieval wiring, auth token
validation, boundary modules that had no test reference before (trace SSE framing and the safe-trace
wrappers, RAG generation argument forwarding and laziness, OSS host/signing/PUT requests, and vision
question construction with image analysis classification), default user seeding without hardcoded
passwords, `.env.example`/`config.py` parity, checkpointer helpers, semantic chunking, chunk-id
namespace isolation between keyword windows and the stored slices, config parsing, grounding
helpers, JSON loading, list-endpoint SQL query-count bounds and message-history pagination, the cost
of keyword recall staying tied to how much content matches rather than to knowledge-base size (issue
#59: the recall SQL must carry a `LIMIT` and must not load files that never matched, asserted
against a knowledge base far larger than `top_k`, and the pre-fix implementation is frozen in the
file and compared item by item on content, order, score and hit evidence), knowledge base name race
handling (concurrent duplicate-name create/rename, session rollback and the global IntegrityError
fallback), IntegrityError diagnosis that branches on which constraint actually fired instead of
reporting a name conflict for every source (issue #83 item 6: the request user's row is deleted
underneath the request with `PRAGMA foreign_keys=ON`, through the real routes and the real global
handler), the `knowledge_files.knowledge_base_id` index without which the per-base count aggregation
still scans the whole table (issue #83 item 5: the with-index and without-index plans are pinned
side by side via `EXPLAIN QUERY PLAN`, plus the migration-side backfill for databases that predate
it), knowledge ownership filtering, knowledge deletion ordering, the reclamation of derived files
when their owner goes away (issue #128: deleting a conversation must also delete its attachment
objects from OSS, and replacing an avatar must delete the file it replaced - the OSS assertions stub
`httpx.Client` at the real request egress so the method, URL and signature are checked, and two
guard cases cover a first upload and a path that is its own replacement so the new deletion path
cannot over-delete - the second of the two was a same-second re-upload until issue #186 replaced the
`user_{id}_{timestamp}` template with a random key, which made that scenario unreachable, so it now
calls the removal helper directly and a sibling case pins that two back-to-back uploads no longer
land on one name), knowledge upload type and size limits, PDF text extraction from bytes constructed
in the test file rather than by a PDF-writing dependency (issue #98: the Helvetica/ASCII path and
the CJK path, which goes through a CID font's CMap instead of WinAnsi and which the stubbed
`extract_file_text` in `test_knowledge_service.py` could never reach), learning trace handling,
OpenAI-compatible URL helpers, memory context, the three missing indexes on
`messages.conversation_id`, `conversations.user_id` and `conversations.knowledge_base_id` without
which a page of an old conversation walks back along the primary key until the page is full and the
sidebar list pays a temporary B-tree sort on top of the scan (issue #176: the with-index and
without-index plans are pinned side by side as in issue #83, but the plans are taken from the
statements the ORM really emitted - captured through `before_cursor_execute` and replayed with their
own bound parameters - so a query that changes shape invalidates the assertion instead of quietly
passing; the migration side drives `_ensure_schema_columns()` itself rather than the helper it
calls, and the idempotency clause pins the helper's existing-index short-circuit by requiring the
second call to log nothing, since deleting that short-circuit and letting `CREATE INDEX` fail into
the swallowed-exception path leaves the index count unchanged as well), Milvus client behavior,
RAGAS text handling, rerank fallback, SECRET_KEY startup guard, MYSQL_PASSWORD startup guard (issue
#185: the database password gets the same treatment as SECRET_KEY - an unset or still-placeholder
`MYSQL_PASSWORD` must refuse to start with an actionable message rather than being silently spliced
into `DATABASE_URL`, an explicitly configured value must still pass so the guard cannot be an
unconditional refusal, the placeholder is never the value in use even on a path that skips the gate,
`ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD` allows startup while warning and is proven independent of
`ALLOW_INSECURE_DEFAULT_SECRET` in both directions, and `.env.example` must document a placeholder
instruction rather than a copyable value - see `test_env_example_parity.py`), SSE client-disconnect
session cleanup (a client that goes away must still close the DB session and return the connection
to the pool), internal exception text staying out of what the user receives (issue #96: neither a
learning trace's SSE frames nor an `HTTPException` detail may carry the exception's own text or the
`OperationalError`/`sqlite3` wording that would give the backend away, while the server-side log and
a reportable id survive; the SSE cases run the real `TraceRecorder`, the real `sanitize_trace_value`
and the real frame encoder and stub only persistence, since `sanitize_trace_value` does not mask an
`error` key and a clean frame can therefore only come from the write side never putting the text
there), the built-in secret-scan rule self-test (all five patterns, the exemption-marker boundary -
including a dual-path fixture proving the marker is really consulted for the assignment hit while
still failing to exempt the `sk-` shape hit on the same line - and a mutation check that neutering
any single rule stops its sample from being reported, plus the gitleaks-layer attribution of issue
#168, where a scanner error and a found secret have to be told apart by gitleaks' own finding marker
rather than by its exit code: a scanner error must exit 2 without ever claiming a found secret, a
real hit must still exit 1, and a third case pins the `--source` path reaching the native binary in
a form it can open, with the target directory carrying a space and non-ASCII bytes so a broken
normalisation shows up), stream fallback reset handling, trace CRUD, purging a conversation's
learning trace together with the conversation (issue #127: only that conversation's rows may go -
same-user traces in other conversations must survive, or emptying the table would pass as well - and
both trace read paths must return nothing afterwards, including rows orphaned before the delete and
rows an in-flight stream writes back), trace event `index` values staying self-consistent when a
delete interleaves with a stream that is still finishing (issue #141: the write path had borrowed a
read-side guard, so `get_trace_snapshot` returned `None` for a row it deemed unreadable and the
caller restarted the numbering at 1; the cases pin the numbering in both the `mid_stream` and
`after_finish` arms and require that the fix not buy that consistency by loosening the read guard
behind #127's 404, which must keep returning `None` for a row it deems unreadable), upload
validation, and reporting an upload whose text extraction produced nothing (issue #166: a PDF with
no text layer used to return 200 while holding zero vectors and no stored source text, so the row
could never be retrieved and had no backfill path - the cases drive the real upload route and reject
it with a dedicated 400 before the metadata row and the vectors are written, pin
`chunk_coverage_ratio` at 0.0 on an empty source so the existing low-coverage guard stops
short-circuiting on exactly the input that loses everything, and extend the empty-content preview
placeholder from `.docx` to `.pdf`/`.txt`/`.md`; the no-text PDF is built from bytes in the test
file, matching `test_pdf_extraction.py`), and the knowledge-base delete path's file listing no
longer instantiating whole rows (issue #179: `list_files_for_knowledge_base` carried the same root
cause #61 fixed only in its sibling `list_knowledge_files`, so both callers - the one that collects
ids to clean vectors and the one that only deletes rows - were pulling the LONGTEXT `content`
column; the cases assert the projected column list equals the sibling's five metadata columns
instead of counting statements, and the mutation matrix shows both directions matter, since a column
set widened with `content` and one narrowed by a missing `created_at` each turn it red, while
`defer` - which also stops the content read - is red too because it leaves six columns; the same
file pins that the per-row `db.delete()` on partially loaded rows still emits a primary-key DELETE
and adds no compensating per-id content read, and pins the one content read that remains, the ORM's
cascade load on `KnowledgeBase.files` at `db.delete(entry)`, so it can neither be mistaken for this
function's query nor change shape unnoticed), and the vision outbound exit sharing one `object_key`
shape predicate with the write and delete exits (issue #181: the attachment column is echoed back by
the client verbatim, so a key like `rag-chat/../../../finance-archive/secret.png` was quoted
straight into a URL handed to the third-party vision service - the `..` survives `quote(key,
safe="/")` and httpx normalises the path - while the write exit dropped it and the delete exit
refused to sign it; the guard now sits in `_public_oss_url`, the single place a bucket URL is built,
so a future caller inherits it too, and the cases assert at the request egress rather than on a
helper's return value: the vision `httpx.AsyncClient` is replaced by a recorder and its record must
stay empty, with positive controls pinning that a minted key still builds its URL and still sends
the request, that a mixed attachment list drops only the foreign entry, and that the
URL-construction probe itself fires so the "zero constructions" assertions cannot pass vacuously),
and token revocation after logout or a password change (issue #184: the server had no way to
invalidate an already-issued token at all - `logout` returned `{"message": "ok"}` without touching
any state, and a token signed before `PUT /api/user/password` kept working for the rest of its 24h
lifetime, so changing a leaked password left the leaked session online; the fix carries two
independent handles in the payload, `ver` (the user's generation, bumped in the same transaction as
the new hash) and `jti` (that one token), because a single handle cannot serve both cases -
generation alone makes one logout kick every device the user is signed in on, and `jti` alone cannot
invalidate a whole batch at once; the cases drive the real `auth_router` and `user_router` over
in-memory SQLite and assert on the **status code of protected endpoints** rather than on whether a
row exists, since a revocation set that is recorded but never consulted by the validation chain
stays green - and both authentication chains are covered, the dependency-injected `get_current_user`
and the streaming entry point that opens its own session and used to call `decode_token` directly,
which is where a half-done fix leaves the old token accepted; the two-round case pins that the
generation advances on every change rather than only the first, positive controls pin that fresh
tokens, other sessions of the same user and other users are untouched, a token forged without the
generation claim is rejected so the fix is not silently scoped to new tokens only, and the streaming
entry point is shown to return its connection to a size-1 pool when authentication fails; the
mutation matrix records that stripping the four revocation semantics turns eleven of the fifteen
cases red while the four that stay green are exactly the positive controls, and that reverting only
the streaming chain to its old shape turns all three streaming cases red), and request throttling at
the two entry points that had none (issue #183: the login route counts failures per account + source
address in a sliding window and answers 429 with `Retry-After` once the threshold is crossed, while
a correct password inside the window still returns 200 - the positive control that keeps the fix
from being a hair-trigger lockout - and the lockout is proven to expire by advancing an injected
clock instead of sleeping, is scoped to the key that actually failed so a second account is
untouched, and is keyed on the socket peer rather than `X-Forwarded-For` so a forged header cannot
buy a fresh bucket; `POST /api/chat/stream` takes a non-blocking concurrency slot after
authentication and refuses the request past the limit rather than queueing it, driven by a hanging
fake upstream: the rejected request must return before the upstream is released, the stream that
holds the slot must still run to `[DONE]`, and the request after it must be accepted again - which
is what pins the release on all five teardown paths (normal completion, an exception raised inside
the stream, a failure before the stream starts, the early return that streams the image-analysis
failure, and a client disconnect through `aclose()`); and the password fields carry `max_length` at
bcrypt's 72-byte input size, so an oversized password is 422 instead of reaching the comparison,
with a boundary control that a password exactly at the limit still logs in), and the three list
endpoints that used to answer with everything their owner had (issue #191: `list_knowledge_bases`,
`list_conversations` and `list_knowledge_files` all ended their SQL in `.all()` with no page-size
parameter at any of the three layers - the trade-off #61 took deliberately when it capped only the
sibling message read and deferred the rest to a tracking issue that was promised in PR #81's "later"
section and never opened, which is why the promise has to be quoted from that pull request's body
rather than found in the tree; all three now take a keyset cursor instead of an offset, since an
offset shifts under rows inserted between pages and serves one row twice while hiding another - `id`
ascending for the knowledge bases and descending for the files, and a composite `(updated_at, id)`
for the conversations, whose primary key is a uuid and so cannot order rows while its second-level
timestamp ties within a second, which is also why the timestamp is normalised to second-level text
before it is compared: the two write paths store it in different shapes and an equality against a
Python datetime never holds, and the next page then re-serves exactly the rows just read; the cases
pin the default page against the seeded set, page through to collect every row exactly once with a
same-second arm that only the tie-breaker can order, show a doubled data set still returning one
page, drive all three routes to 422 on an out-of-range limit and on a half-given cursor while a
direct service call is refused rather than silently clamped the way the CRUD layer below it clamps
for its own callers, and pin the probe's criterion against both a positive and a negative sample so
the three reads cannot quietly go uncapped again). The last one is joined by the streaming and
memory paths' synchronous DB work leaving the event loop thread (issue #187: `memory_service.py` and
`chat_service.py` still opened a request-scoped `SessionLocal()` and ran
`query`/`add`/`commit`/`close` on the event loop thread - the AST scan counted 11 and 17 unwrapped
sinks there against 0 in the already-fixed `retrieval.py` (the 17 is a recount, not the original
reading: the scan reported with #187 read 11 and 18, and the #184 auth refactor then replaced one of
those bare `db.query()` calls with `authenticate(db, ...)`) - so one network DB round trip froze
every concurrent request's token stream in the same process; the cases reuse the heartbeat oracle
this repository already settled on for ingest and retrieval, with a ticker coroutine that must be
scheduled while the call runs, which is 0 both before the fix and with the fix reverted out of the
function under test, and they pin the thread rule directly rather than by timing: each Session
records the thread it was opened, used and closed on, so a session handed to another thread turns
red even when the timings look fine - which is also why the keyword recall now opens its own session
when the caller has none, since handing `stream_chat`'s session to that worker is exactly what the
issue forbids; both the entry `SessionLocal()` + user lookup and the in-stream assistant write are
covered, the write steps are shown to close their own session when they raise, and the scan itself
is a gate with 2 positive and 3 negative self-check cases so a zero cannot pass vacuously. Two
existing assertions moved with the fix and are recorded here rather than quietly dropped:
`stream_chat` no longer passes a Session to `retrieve_knowledge` (the new assertion is `is None`,
and any revival of the old hand-off turns it red), and a streaming request no longer holds a pooled
connection for the whole stream, so the precondition `test_sse_session_leak.py` used to assert at 1 -
itself a leak detector - now asserts at 0 from the same property. The same oracle then reaches the
learning trace's own writeback, which #187 left outside its boundary (issue #201: with
`LEARNING_TRACE_ENABLED` at its production default of true, `TraceRecorder`'s
`__init__`/`add`/`attach`/`finish` opened their own synchronous `SessionLocal()` through
`crud/trace.py` on whatever thread called them, which in `stream_chat`/`event_stream` is the event
loop thread - measured on a copy of the pre-fix tree at 10 sync statements, calling thread identical
to the loop thread, and 0 heartbeats in a 1521ms window; `add`/`attach`/`finish` are coroutines now
and each hands its whole "open session, use it, close it" step to `asyncio.to_thread`, the
constructor no longer writes because a constructor cannot be awaited and `persist_trace_session`
already upserts, and the module-level `append_trace_event` - which `memory_service` and `ragas_eval`
call from coroutines - is wrapped the same way at those call sites while the sites that already run
in a worker thread or with no loop at all keep calling it directly; the cases intercept
`crud.trace.SessionLocal` rather than `chat_service.SessionLocal`, because that module owns the
session the previous gates could not see, and they run the real `stream_chat` end to end with the
trace switched on rather than stubbed out, which is what turns red if only `_persist` is moved and
the `add`/`attach`/`finish` chain is left behind; ordering and terminal state are pinned against the
in-memory events after every await, so a fire-and-forget rewrite that returns before the row is
written fails too, and the disconnect path is driven through `aclose()` to pin that the teardown
write in the stream generator's `finally` still lands; six mutations were each reverted in an
out-of-repo copy against a green control - the recorder's persist back on the loop, the constructor
persisting again, `add` writing inline, one `memory_service` call site unwrapped, and each
`_trace_add` helper writing inline - and every one turns the cases that name it red, with the two
helper modules split into separate cases so that a single one regressing cannot hide behind the
other's worker-thread sessions; the scan is a second gate over the five modules that reach the trace
chain, with 2 positive and 5 negative self-checks, and a companion case pins that every `crud_trace`
call in `learning_trace.py` sits in a synchronous payload while the recorder's write methods are
coroutines), and the retired `/api/checkpointer/threads` endpoint (issue #236: it had answered
`{"threads": []}` since the day it was registered, because the `checkpoints` table has no writer
anywhere in the request paths and a `thread_id` cannot be attributed to a user at all - so the
endpoint carried no content and no access control to speak of, and the trade-off taken is removal
rather than inventing an owner filter over a mapping nothing in the tree confirms; the cases pin the
path out of `main.app.openapi()["paths"]` rather than out of a walk over `main.app.routes`, since
`include_router` does not flatten child routers and that walk returns no `/api` path at all, which
would make the assertion vacuously true; the 404 is taken from a request against the real `main.app`
with a sibling `/api` route as the positive control that the whole app is still up, and a reverse
lock keeps `save_checkpoint`/`load_checkpoint`/`delete_thread_checkpoints` callable, so removing the
endpoint cannot pass by removing the module). `test_message_feedback_250.py` covers the
message-level like/dislike feedback added by issue #250: `POST
/api/chat/messages/{message_id}/feedback` persists one of `1`/`-1`/`0` into the new
`messages.feedback` column and echoes it back in conversation history (T1), re-sending a value is
idempotent and leaves the row's `created_at` and row count untouched because the write is an
in-place ORM assign plus `commit`/`refresh` rather than a bulk `update` (T2, which the shared
`FakeDb` cannot support anyway), acting on someone else's message or an unknown id is a `404` with
no content echoed and the stored value unchanged (T3/T4), a value outside `Literal[-1, 0, 1]` or a
missing `feedback` key is a `422` whose `detail[0]["type"]` is `literal_error`/`missing` (T5),
feedback on a user-authored message is rejected by the assistant-only role guard (T6), a freshly
created message defaults to `0` (T7), and `_ensure_schema_columns()` adds the column to a legacy
table idempotently so the pre-existing row reads back as `0` (T8). Conversation renaming rejects
whitespace-only titles without changing storage and trims valid titles while preserving internal
spaces, the request length limit, authentication and ownership checks (#265). Conversation Markdown
export (#252) reads real SQLite messages in ascending ID batches of at most 200, without changing
chat-history paging. Authenticated HTTP cases cover complete exports above the page cap, tied
timestamps, ID gaps, user ownership, source-field filtering, empty conversations, safe UTF-8
filenames, a fixed initial upper ID, and failure before any partial download can be returned. Model
generation that normally exhausts without any final text sends one explicit error before DONE,
records `generation_failed` with `empty_response` and `assistant_not_saved`, and finishes the failed
trace before the client can disconnect on that error (#259). The cases replace only model factories
along the real chat-service/chain/LLM path, cover zero chunks and all-empty chunks, and keep
empty-prefix success, existing model errors and fallback reset semantics intact. Cancellation,
direct generator exceptions and pre-generation rejection keep their existing paths; no empty
assistant is stored, no automatic retry is introduced, and the same conversation accepts the next
question after the failed stream releases its concurrency slot. `test_eval_harness_293.py` pins the
offline RAG evaluation harness (issue #293): the repository fixture loads and validates, the six
required categories each contribute two cases plus twenty exact-term entries reserved for #288,
missing fields and duplicate IDs are reported without suppressing the report, external question sets
stay path-private and warn instead of failing on absent categories, two dry runs are identical apart
from the timestamp, Recall@5/20 is computed only from explicitly supplied retrieval results, and the
recorded character-chunk baseline is checked against `chunk_text`'s AST defaults. `test_bm25_288.py`
covers issue #288's BM25 keyword ranking: tokenisation, NFKC folding, length normalisation, document
frequency from every knowledge-base window rather than the LIKE candidate set, the candidate budget
that caps full scoring at `BM25_MAX_SCORED_CANDIDATES` (asserted by counting `score_text` calls,
never by wall clock), the vocabulary cap, stable cache keys with single-flight rebuilds, prewarm,
the no-candidate/no-index path, and a 20-question exact-term fixture with realistic paragraphs and
near misses - the legacy ranking finds none of the targets in its top five while BM25 ranks all
twenty first.

`conftest.py` puts `backend/` on `sys.path` so the tests can import application modules, and holds the test doubles shared by more than one test file: the `FakeQuery`/`FakeDb`/`FakeUser`/`FakeKnowledgeBase`/`FakeTraceRecorder` classes, the pytest fixtures built on them (`fake_user`, `fake_db`, `fake_knowledge_base`, `trace_recorder_cls`), and the SSE helpers (`collect_stream`, `parse_sse_frames`, `frames_of_type`, `streamed_content`). Test doubles used by a single file stay in that file.

`test_milvus_acceptance.py`, `test_retrieval_acceptance.py` and `test_provider_smoke.py` are the behavioral acceptance layer for the maintenance goals in `docs/MAINTENANCE_GOAL_CLOSURE.md`: a real Milvus Lite round trip (upload, query, delete, knowledge-base isolation, index rebuild), the full `retrieve_knowledge` chain (multi-route recall, RRF fusion, rerank, final context order), and the provider clients (DeepSeek, embedding, rerank and their fallbacks) including failure, timeout and malformed-response paths. They need no network and no credentials.

The provider smoke entry point is `scripts/smoke_providers.py`; it runs offline against in-process stubs by default and only reaches the real endpoints with `--live`.

## Excluded Local Files

Do not stage generated or design-only files:

- `__pycache__/`
- `*.pyc`
- `.pytest_cache/`
- `tests/.test.md`
- `tests/*.test.md`

The ignore rules are intentionally kept in the repository `.gitignore` so the executable test baseline can be staged without cache files.
