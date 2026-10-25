# 部署指南（配置 / 构建 / Nginx / DNS / HTTPS）

本文件是生产部署的完整细节：环境变量、数据库初始化、构建启动与反向代理。本地运行与依赖安装见 [README 快速开始](../README.md#快速开始)。

## 生产配置

完整配置以仓库根目录的 [`.env.example`](../.env.example) 为唯一事实来源。后端启动时从仓库根目录读取 `.env`，随后读取 `.env.development`；后读的文件不会覆盖 `.env` 中已经存在的同名变量。

复制示例配置：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

至少需要配置：

```env
MYSQL_USER=your_mysql_user
MYSQL_PASSWORD=your_mysql_password
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_DATABASE=rag_system
MYSQL_SSL_MODE=
MYSQL_SSL_CA=

DEEPSEEK_API_KEY=your_deepseek_api_key
DASHSCOPE_API_KEY=your_dashscope_api_key

SECRET_KEY=change-this-secret-key-in-production
```

上面 `SECRET_KEY` 写的是与 `.env.example` 一致的公开占位值，**必须替换**成随机值再启动：保持占位值（或留空）时后端会拒绝启动。生成真实值：

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

`MYSQL_PASSWORD` 同样不能留空，也不能继续使用仓库公开的占位值 `change-me`，否则启动阶段会拒绝连接。生成数据库口令，并把同一口令设置到上面配置的 MySQL 账号：

```bash
python -c "import secrets; print(secrets.token_urlsafe(24))"
```

仅本地开发可以分别用 `ALLOW_INSECURE_DEFAULT_SECRET=true`、`ALLOW_INSECURE_DEFAULT_MYSQL_PASSWORD=true` 临时放行；两个开关相互独立，开启一个不会放宽另一个的检查，且都禁止用于任何他人可访问的部署。

特殊场景按需追加：

- Render 后端 Web Service 建议额外设置 Python 版本，避免平台默认使用过新的 Python 版本：`PYTHON_VERSION=3.10.11`。
- 使用 Aiven MySQL 这类要求 SSL 的云数据库时，在部署平台（如 Render）的环境变量里加 `MYSQL_SSL_MODE=required`；服务商要求指定 CA 证书时，把证书文件随部署环境挂载后设置 `MYSQL_SSL_CA=/path/to/ca.pem`。
- 本地 Milvus Lite：`MILVUS_LITE_URI=./milvus.db`。`MILVUS_URI` 默认与它相同，**不要留空**：留空会让运行时拿到空地址，覆盖对 `MILVUS_LITE_URI` 的回退；修改其中一个时请同步另一个。
- 远程 Milvus：`MILVUS_URI=http://your-milvus-host:19530`，按需补充 `MILVUS_TOKEN`、`MILVUS_USER`、`MILVUS_PASSWORD`、`MILVUS_DB_NAME`。
- 离线/受限网络部署：入库分批用 tiktoken 的 `cl100k_base` 估算 token，首次使用会尝试下载 BPE 文件。请先在一台可联网的机器上生成缓存目录，再随部署环境挂载，并把目录路径写进 `TIKTOKEN_CACHE_DIR`：

  ```bash
  TIKTOKEN_CACHE_DIR=/path/to/tiktoken-cache python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"  # scan-secrets:allow a cache path, not a credential
  ```

  启动日志会打印实际生效的缓存目录与缓存文件名（`9b5ad71b2ce5302211f9c61530b329a4922fc6a4`，即 `cl100k_base.tiktoken`）。该检查失败不会阻止服务启动，但上传入库会因 token 估算失败而报错并写明缓存目录。仓库不提交 BPE 文件。
- 图片附件依赖阿里云 OSS：需要配置 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`、`OSS_BUCKET`、`OSS_ENDPOINT`；未使用图片附件时可以留空。
- RAGAS 是可选在线评估能力，默认 `RAGAS_ENABLED=false`；启用前需额外安装 `backend/requirements-ragas.txt`。

默认账号播种由 `SEED_DEFAULT_USERS` / `SEED_ADMIN_PASSWORD` / `SEED_DEMO_PASSWORD` 控制：

- `SEED_DEFAULT_USERS` 默认为 `true`，设为 `false` 后启动过程不会创建任何账号；为 `true` 时仅为尚不存在的 `admin`、`demo` 创建账号，且不会覆盖已有账号的口令。
- 账号口令只来自 `SEED_ADMIN_PASSWORD` / `SEED_DEMO_PASSWORD`，仓库中不存在固定口令；未配置对应变量时会生成不可预测的随机口令，随机口令不会写入日志。口令两端的空白字符会被忽略。
- 因此未显式配置口令的账号无法直接登录，**已存在**的账号可以按下面的方式重置口令（`新口令` 换成自定义值）：

```bash
cd backend
python -c "from database.session import SessionLocal; from crud import user as crud_user; from model.models import User; from service.auth_service import pwd_context; db = SessionLocal(); u = db.query(User).filter_by(username='admin').first(); crud_user.update_password_hash(db, u, pwd_context.hash('新口令')); db.close(); print('password updated')"
```

- 如果该账号还不存在（例如长期设置 `SEED_DEFAULT_USERS=false`，库里没有任何账号），临时把 `SEED_DEFAULT_USERS` 设回 `true` 并配置 `SEED_ADMIN_PASSWORD`，重启服务一次即可创建；播种只补建不存在的账号，不会覆盖已有账号的口令。
- 对外提供服务前，建议设置 `SEED_DEFAULT_USERS=false`，或至少为启用的账号配置强口令。

## 初始化数据库与存量回填

创建 MySQL 数据库：

```sql
CREATE DATABASE IF NOT EXISTS rag_system
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;
```

启动 FastAPI 时，`backend/main.py` 会调用 `init_db()` 创建表结构，并调用 `seed_default_users()` 初始化默认账号（播种规则见上一节）。

知识库与知识文件按归属用户隔离，历史数据（`user_id` 为 NULL）对任何用户都不可见，需要一次性回填：

```bash
python scripts/backfill_knowledge_owner.py --user-id <用户ID>            # 默认只打印回填计划
python scripts/backfill_knowledge_owner.py --user-id <用户ID> --apply    # 真正写入
```

先不带 `--apply` 执行，确认计划中的归属用户与记录数无误后，再加 `--apply` 真正写入。

## 生产构建与启动

构建前端：

```bash
npm run build
```

构建产物位于：

```text
dist/
```

前端请求的后端地址在**构建期**由 `VITE_API_BASE_URL` 固化，生产值必须与 Nginx 的代理路径一致（同源部署保持 `/api`）：

```bash
VITE_API_BASE_URL=/api npm run build
```

后端可使用 Uvicorn / Gunicorn + Uvicorn Worker 运行。例如：

```bash
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8002
```

## Nginx 反向代理

下面示例假设：

- 域名：`example.com`
- 前端静态文件目录：`/var/www/rag/dist`
- 后端地址：`http://127.0.0.1:8002`

```nginx
server {
    listen 80;
    server_name example.com;

    root /var/www/rag/dist;
    index index.html;

    location = /health {
        proxy_pass http://127.0.0.1:8002/health;
    }

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://127.0.0.1:8002/api/;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_buffering off;
        proxy_read_timeout 300s;
    }

    location /uploads/ {
        proxy_pass http://127.0.0.1:8002/uploads/;
    }
}
```

`proxy_buffering off` 对 SSE 很重要，否则流式回答可能被 Nginx 缓冲，导致前端不能实时显示。

健康检查必须单独代理：后端的健康检查路由注册在**根路径 `/health`**，不在 `/api/` 前缀下，所以上面的 `location = /health` 不能省。省略它的话，`/health` 会落进 `location /` 的 `try_files`，返回前端 `index.html`（`200` + `text/html`）而不是健康检查的 JSON，部署自检就会误判。另外**`/api/health` 并不存在**，请求它只会得到 `404`。

关于接口文档：本示例**故意不代理** `/docs`、`/redoc`、`/openapi.json`。这三个是 FastAPI 挂在根路径下的交互文档与 OpenAPI Schema，按上述配置在公网不可达（会被 `location /` 兜到前端页面），可以避免对外暴露完整的接口结构。如果确实需要在受控环境里访问，在 server 块内补充：

```nginx
    # Swagger UI 页面本身是 /docs，它还会请求 /docs/oauth2-redirect，
    # 因此精确匹配 /docs 之外还要放行 /docs/ 子路径。
    # proxy_set_header 在 location 之间互不继承（写在 server 级才会被各 location 继承），
    # 这里逐块书写，方便单独复制其中一段。
    location = /docs {
        proxy_pass http://127.0.0.1:8002/docs;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /docs/ {
        proxy_pass http://127.0.0.1:8002/docs/;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location = /redoc {
        proxy_pass http://127.0.0.1:8002/redoc;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location = /openapi.json {
        proxy_pass http://127.0.0.1:8002/openapi.json;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
```

这里统一用 `=` 精确匹配，避免 `/docsXYZ` 这类并不存在的路径也被转发到后端。`proxy_set_header Host $host;` 同样是必须的：不写的话 Nginx 默认把 `Host` 设成上游地址（`127.0.0.1:8002`），而 Starlette 对 `/docs/` 会回一个 `307` 跳转到 `/docs`，跳转目标里就会带上 `http://127.0.0.1:8002` 这个只在服务器内部可达的地址，浏览器拿到后会跳不过去。四个块里还一并设置了 `X-Forwarded-Proto`（与上面 `/api/` 块保持一致），这样从 HTTPS 入口访问时 `307` 直接跳到 `https://`；少了它，跳转目标会是 `http://`，还要经 80 端口再 `301` 回 443，白多一次往返。开启前建议配合 IP 白名单（`allow` / `deny`）或额外的鉴权，不要直接暴露在公网。

## DNS 配置

如果要绑定域名，需要在域名服务商处添加解析记录：

| 类型 | 主机记录 | 指向 |
| --- | --- | --- |
| A | `@` | 服务器公网 IPv4 |
| A | `www` | 服务器公网 IPv4 |
| CNAME | `api` | 可选，指向主域名或后端网关域名 |

配置后可用以下命令检查解析：

```bash
nslookup example.com
```

或：

```bash
dig example.com
```

DNS 生效可能需要几分钟到数小时，取决于 TTL 和域名服务商。

## HTTPS 配置

生产环境建议使用 HTTPS。可以通过 Let's Encrypt 申请免费证书：

```bash
sudo certbot --nginx -d example.com -d www.example.com
```

证书签发后，确认 Nginx 中存在 443 配置，并将 HTTP 自动跳转到 HTTPS：

```nginx
server {
    listen 80;
    server_name example.com www.example.com;
    return 301 https://$host$request_uri;
}
```

上面是 HTTP → HTTPS 跳转。`certbot --nginx` 是**原地改写**匹配到 `example.com` 的那个 server 块（也就是上文「Nginx 反向代理」里监听 80 的那一个），所以那里的 `location = /health` 通常会被一并保留；但如果你另行编写了 443 的 server 块，或换用了其它证书签发方式，就必须逐条确认 443 块里也有这条规则，否则 HTTPS 入口的健康检查仍会被 `location /` 兜成前端页面：

```nginx
server {
    listen 443 ssl;
    server_name example.com;
    # ssl_certificate / ssl_certificate_key 等由 certbot 写入

    location = /health {
        proxy_pass http://127.0.0.1:8002/health;
    }

    # 其余 location / 与 location /api/ 的配置同「Nginx 反向代理」
}
```

HTTPS 部署后需要检查：

- `https://example.com` 可以打开前端页面。
- `https://example.com/health` 可以正常返回健康检查 JSON。
- `/api/chat/stream` 流式输出不会被代理缓冲。
- `VITE_API_BASE_URL` 与 Nginx 代理路径一致。
- 生产环境 `SECRET_KEY` 已替换为强随机值（未替换时后端会拒绝启动）。
- CORS、Cookie、安全响应头按真实部署域名收紧。

其中健康检查这一项可以直接用命令验证：

```bash
curl -i https://example.com/health
```

期望结果是 `200`、`content-type: application/json`，响应体为 `{"status":"ok"}`。如果拿到的是 `text/html`，说明 `/health` 被 `location /` 兜到了前端页面（缺少上文「Nginx 反向代理」的 `location = /health`）；如果拿到 `404`，则是请求了并不存在的 `/api/health`。
