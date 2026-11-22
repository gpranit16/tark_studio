# TARK Studio — Production Deployment Guide
**Zero-Cost Split Production Architecture: Cloudflare Pages + Oracle Cloud Always Free (ARM64) + DuckDNS + Caddy**

---

## 1. Target Architecture Overview

```
                      +---------------------------------------+
                      |   Cloudflare Pages (Free Tier)        |
                      |   - Static React 18 + Vite Frontend   |
                      |   - Global Anycast CDN & Edge Caching |
                      |   - URL: https://tark-studio.pages.dev|
                      +-------------------+-------------------+
                                          |
                        HTTPS (REST + SSE Streaming)
                                          |
                                          v
+-----------------------------------------------------------------------------------+
| Oracle Cloud Always Free Ampere A1 VM (Ubuntu 22.04/24.04 ARM64)                  |
| Specs: 2 OCPU, 12 GB RAM, 200 GB NVMe Storage (100% Free Forever)                 |
| Dynamic DNS: your-subdomain.duckdns.org                                           |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  | Caddy 2 Reverse Proxy (Automatic Let's Encrypt / ZeroSSL TLS)               |  |
|  | - Port 80 / 443                                                             |  |
|  | - flush_interval -1 (Zero-buffering real-time SSE token streaming)          |  |
|  +-------------------------------------+---------------------------------------+  |
|                                        | internal network (bridge)                |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  | TARK Studio Backend (FastAPI + LangGraph)                                   |  |
|  | - Port 8000 (Internal)                                                      |  |
|  | - Multi-turn conversational memory & PostgreSQL checkpointing               |  |
|  | - Embedded Xenova/ms-marco-MiniLM-L-6-v2 ONNX INT8 Cross-Encoder             |  |
|  | - Groq API Integration (LLM Inference)                                      |  |
|  +-------------------+---------------------------------+-----------------------+  |
|                      |                                 |                          |
|                      v                                 v                          |
|  +-----------------------------------+  +--------------------------------------+  |
|  | PostgreSQL 16 + pgvector          |  | Ollama Service                       |  |
|  | - Port 5432 (Internal)            |  | - Port 11434 (Internal)              |  |
|  | - Full-text search (tsvector)     |  | - nomic-embed-text (768d embeddings) |  |
|  | - HNSW / Cosine semantic search   |  | - Auto-pulled on first boot          |  |
|  | - Volume: tark_postgres_data      |  | - Volume: tark_ollama_data           |  |
|  +-----------------------------------+  +--------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Oracle Cloud Always Free VM Setup

### Step 2.1: Provision the Ampere A1 Compute Instance
1. Log into your [Oracle Cloud Console](https://cloud.oracle.com).
2. Navigate to **Compute** > **Instances** > **Create Instance**.
3. **Image and Shape**:
   - Image: **Ubuntu 22.04 LTS** or **Ubuntu 24.04 LTS** (Minimal/Standard aarch64).
   - Shape: **Ampere VM.Standard.A1.Flex** (ARM64).
   - Allocation: **2 OCPU**, **12 GB RAM** (well within the Always Free limit of 4 OCPU / 24 GB).
4. **Networking**:
   - Assign a public IPv4 address.
5. **Boot Volume**:
   - Set Boot Volume to **100 GB** or **200 GB** (Always Free tier includes up to 200 GB free block storage).
6. **SSH Keys**:
   - Paste or download your public SSH key.
7. Click **Create** and note the **Public IP Address**.

---

### Step 2.2: Configure Oracle Cloud Firewall & Ingress Rules
Oracle Cloud requires opening ports in both the Virtual Cloud Network (VCN) Security List and the Ubuntu host firewall.

#### In Oracle VCN Console:
1. Go to **Networking** > **Virtual Cloud Networks** > Click your VCN > Click **Default Security List**.
2. Click **Add Ingress Rules**:
   - **Source CIDR**: `0.0.0.0/0`
   - **IP Protocol**: `TCP`
   - **Destination Port Range**: `80, 443`
   - **Description**: `HTTP and HTTPS for Caddy Reverse Proxy`
3. Click **Add Ingress Rules**.

#### In the Ubuntu VM (via SSH):
SSH into your instance:
```bash
ssh -i /path/to/your-key ubuntu@<YOUR_ORACLE_PUBLIC_IP>
```
Open ports on the local host firewall:
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save || sudo apt-get install -y iptables-persistent && sudo netfilter-persistent save
```

---

## 3. Free DuckDNS Subdomain Setup

1. Go to [DuckDNS](https://www.duckdns.org) and sign in (GitHub, Google, etc.).
2. Create a free subdomain: e.g. `tark-studio.duckdns.org`.
3. Set the IP address to your **Oracle Cloud VM Public IP**.
4. Set up an automated DuckDNS updater cron job on the Ubuntu VM so your IP never goes out of sync:
```bash
mkdir -p ~/duckdns
cat << 'EOF' > ~/duckdns/duck.sh
#!/bin/sh
echo url="https://www.duckdns.org/update?domains=YOUR_SUBDOMAIN&token=YOUR_DUCKDNS_TOKEN&ip=" | curl -k -o ~/duckdns/duck.log -K -
EOF
chmod 700 ~/duckdns/duck.sh

# Add to crontab to run every 30 minutes
(crontab -l 2>/dev/null; echo "*/30 * * * * ~/duckdns/duck.sh >/dev/null 2>&1") | crontab -
```

---

## 4. Install Docker on Oracle Ubuntu (ARM64)

Run the following commands on the Ubuntu VM:
```bash
# Update packages
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker official GPG key & repository for Ubuntu ARM64
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine and Docker Compose Plugin
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Allow ubuntu user to run docker
sudo usermod -aG docker ubuntu
newgrp docker
```

---

## 5. Deploy TARK Studio Backend on Oracle VM

1. Clone your GitHub repository on the Ubuntu VM:
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/tark-studio.git
cd tark-studio
```

2. Configure `.env.production`:
```bash
cp .env.production.example .env.production
nano .env.production
```
Configure your actual values:
```ini
POSTGRES_USER=tark_user
POSTGRES_PASSWORD=generate_a_strong_password_here
POSTGRES_DB=tark_studio

GROQ_API_KEY=gsk_your_actual_groq_api_key

DOMAIN=tark-studio.duckdns.org
ACME_EMAIL=your-real-email@gmail.com

CORS_ORIGINS=https://tark-studio.pages.dev
```

3. Launch the production stack with Docker Compose:
```bash
docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build
```

4. Verify all 4 containers are running and healthy:
```bash
docker compose -f docker-compose.prod.yml ps
```
You should see:
- `tark_db` (healthy)
- `tark_ollama` (running)
- `tark_backend` (healthy)
- `tark_caddy` (running)

5. Test HTTPS health endpoint from any machine:
```bash
curl https://tark-studio.duckdns.org/api/v1/health
```
Response:
```json
{
  "status": "healthy",
  "app_env": "production",
  "database": {"connected": true, "has_vector_extension": true},
  "providers": {"groq_configured": true, "ollama_base_url": "http://ollama:11434"}
}
```

---

## 6. Deploy Frontend on Cloudflare Pages (Free)

1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com/) > **Compute (Workers & Pages)** > **Create application** > **Pages** > **Connect to Git**.
2. Select your `tark-studio` GitHub repository.
3. Configure the build settings:
   - **Project name**: `tark-studio` (or any preferred name, yields `https://tark-studio.pages.dev`)
   - **Framework preset**: `Vite`
   - **Root directory**: `frontend`
   - **Build command**: `npm run build`
   - **Build output directory**: `dist`
4. **Environment Variables**:
   Add:
   - `VITE_API_BASE_URL`: `https://tark-studio.duckdns.org`
5. Click **Save and Deploy**.
6. Cloudflare Pages builds and deploys your application globally in ~40 seconds.

---

## 7. Verification & Production Checklist

- [x] **HTTPS TLS Certificate**: Automatically provisioned by Caddy for `https://tark-studio.duckdns.org`.
- [x] **Real-time SSE Streaming**: Verified with Caddy's `flush_interval -1` setting (tokens stream with zero latency).
- [x] **CORS Resolution**: Cloudflare Pages origin allowed by backend CORS configuration.
- [x] **ARM64 Native Execution**: Tested on Ubuntu aarch64 with native wheels (ONNX Runtime, asyncpg, PyMuPDF).
- [x] **Persistent Storage**: Named volumes preserve PostgreSQL data, uploaded PDFs, Ollama embeddings, and Caddy TLS certificates across container restarts and VM reboots.

---

## 8. Cloud-Native Production Deployment (Render + Neon + Cloudflare Workers AI)

For fully autonomous cloud deployments (zero dependence on a local PC, local Ollama, or personal machine):

### Architecture
- **Frontend**: Cloudflare Pages (`https://tark-studio.pages.dev`)
- **Backend**: Render Web Service (Docker runtime)
- **Database**: Neon Serverless PostgreSQL 16+ with `pgvector`
- **Embeddings**: Cloudflare Workers AI (`@cf/google/embeddinggemma-2`, 768 dimensions)
- **Reranker**: Xenova INT8 Cross-Encoder (`Xenova/ms-marco-MiniLM-L-6-v2`)
- **LLM**: Groq API (`llama-3.3-70b-versatile`)

### Render Environment Variables
```env
APP_ENV=production
DATABASE_URL=postgresql://neondb_owner:npg_eCd8XkZUMG5j@ep-cold-river-becd7pf3-pooler.c-9.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require
GROQ_API_KEY=gsk_...
DEFAULT_PROVIDER=groq
DEFAULT_MODEL=llama-3.3-70b-versatile
CORS_ORIGINS=https://tark-studio.pages.dev,http://localhost:5173,http://localhost:3000
HOST=0.0.0.0
APP_PORT=10000

# Cloudflare Workers AI Embeddings (Independent Cloud Pipeline)
EMBEDDING_PROVIDER=cloudflare
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token
CLOUDFLARE_EMBEDDING_MODEL=@cf/google/embeddinggemma-2
```
