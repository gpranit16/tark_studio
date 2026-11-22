# TARK Studio — Data Migration, Backup & Restoration Guide

This guide provides step-by-step procedures to migrate your local conversations, ingested documents, vector embeddings, full-text indexes, and LangGraph checkpoints to your production Oracle Cloud Always Free VM, and to manage ongoing automated backups.

---

## 1. Information Preserved During Migration

When migrating or backing up TARK Studio, the following state is preserved:
1. **PostgreSQL Database**:
   - `conversations`: Thread titles, metadata, archive status, and model selections.
   - `messages`: Full message history with latency metadata, token counts, and citations.
   - `documents`: File records, hash digests, page counts, chunk counts, extracted links & tables.
   - `document_chunks`: Extracted text, viewer/book page numbers, chunk indexes, FTS tsvectors, and 768d vector embeddings.
   - `checkpoints`, `checkpoint_writes`, `checkpoint_blobs`: LangGraph stateful multi-turn session checkpoints.
2. **Uploaded Document Files**:
   - Original PDF/text/markdown files in `backend/uploads/` (referenced by `Document.id`).

---

## 2. Step-by-Step Data Export from Local Environment

### Step 2.1: Dump Local PostgreSQL Database
Run this command from your local machine to export a compressed PostgreSQL dump:

**If running PostgreSQL in Docker (`tark_pgvector`):**
```bash
docker exec -t tark_pgvector pg_dump -U postgres -d tark_studio -Fc -f /tmp/tark_studio_backup.dump
docker cp tark_pgvector:/tmp/tark_studio_backup.dump ./tark_studio_backup.dump
```

**If running local PostgreSQL natively:**
```bash
pg_dump -h localhost -p 5433 -U postgres -d tark_studio -Fc -f tark_studio_backup.dump
```

---

### Step 2.2: Archive Local Uploaded Documents
Archive your uploaded documents folder:

**Linux / macOS:**
```bash
tar -czvf tark_uploads.tar.gz -C backend/uploads .
```

**Windows (PowerShell):**
```powershell
tar.exe -czvf tark_uploads.tar.gz -C backend/uploads .
```

---

### Step 2.3: Transfer Backup to Oracle VM
Securely copy the dump and uploads archive to your Oracle VM using `scp`:
```bash
scp -i /path/to/your-key tark_studio_backup.dump tark_uploads.tar.gz ubuntu@<ORACLE_VM_IP>:~/
```

---

## 3. Step-by-Step Data Import on Oracle Cloud VM

Connect to the Oracle VM via SSH:
```bash
ssh -i /path/to/your-key ubuntu@<ORACLE_VM_IP>
```

### Step 3.1: Ensure Production Stack is Running
Make sure the production containers are up:
```bash
cd ~/tark-studio
docker compose -f docker-compose.prod.yml up -d
```

---

### Step 3.2: Restore Database into Production Container
1. Copy the backup file into the running PostgreSQL container (`tark_db`):
```bash
docker cp ~/tark_studio_backup.dump tark_db:/tmp/tark_studio_backup.dump
```

2. Restore the database schema and data using `pg_restore`:
```bash
docker exec -i tark_db pg_restore \
    -U tark_user \
    -d tark_studio \
    --clean \
    --if-exists \
    --no-owner \
    --no-privileges \
    /tmp/tark_studio_backup.dump
```

3. Remove the temporary dump file from the container:
```bash
docker exec -i tark_db rm /tmp/tark_studio_backup.dump
```

---

### Step 3.3: Restore Uploaded Documents Volume
Restore the original uploaded files into the `tark_uploads` persistent volume:
```bash
docker run --rm \
    -v tark_uploads:/dest \
    -v ~/tark_uploads.tar.gz:/src/uploads.tar.gz \
    alpine sh -c "cd /dest && tar -xzvf /src/uploads.tar.gz"
```

---

### Step 3.4: Verify Restored Data Integrity
Run verification queries directly in the PostgreSQL container:
```bash
docker exec -it tark_db psql -U tark_user -d tark_studio -c "
SELECT 
    (SELECT COUNT(*) FROM conversations) AS conversations_count,
    (SELECT COUNT(*) FROM messages) AS messages_count,
    (SELECT COUNT(*) FROM documents) AS documents_count,
    (SELECT COUNT(*) FROM document_chunks) AS chunks_count,
    (SELECT COUNT(*) FROM checkpoints) AS checkpoints_count;
"
```

You should see your exact counts of conversations, messages, documents, chunks, and checkpoints.

---

## 4. Automated Production Backup Strategy (Daily Cron)

To ensure zero data loss on Oracle Cloud, set up an automated daily backup script.

### Create Backup Script on Ubuntu VM:
```bash
mkdir -p ~/backups
cat << 'EOF' > ~/backups/backup_tark.sh
#!/bin/bash
set -e
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/home/ubuntu/backups"
KEEP_DAYS=7

# 1. Dump PostgreSQL
docker exec -t tark_db pg_dump -U tark_user -d tark_studio -Fc -f /tmp/db_${TIMESTAMP}.dump
docker cp tark_db:/tmp/db_${TIMESTAMP}.dump ${BACKUP_DIR}/tark_db_${TIMESTAMP}.dump
docker exec -t tark_db rm /tmp/db_${TIMESTAMP}.dump

# 2. Archive Uploads
docker run --rm -v tark_uploads:/src:ro -v ${BACKUP_DIR}:/dest alpine \
    tar -czf /dest/tark_uploads_${TIMESTAMP}.tar.gz -C /src .

# 3. Clean up backups older than 7 days
find ${BACKUP_DIR} -name "tark_db_*.dump" -mtime +${KEEP_DAYS} -delete
find ${BACKUP_DIR} -name "tark_uploads_*.tar.gz" -mtime +${KEEP_DAYS} -delete

echo "[$(date)] Backup completed: db_${TIMESTAMP}.dump, uploads_${TIMESTAMP}.tar.gz" >> ${BACKUP_DIR}/backup.log
EOF
chmod +x ~/backups/backup_tark.sh
```

### Schedule in Crontab:
```bash
# Run daily at 03:00 AM UTC
(crontab -l 2>/dev/null; echo "0 3 * * * ~/backups/backup_tark.sh") | crontab -
```

---

## 5. Security & Privacy Safeguards

- **Never Commit Backups to Git**: `*.dump`, `*.sql`, `*.tar.gz`, and `*.backup` files are strictly excluded in `.gitignore`.
- **Database Secrets Isolation**: Passwords and connection URLs are kept strictly in `.env.production` (never tracked).
- **Untrusted File Ingestion**: Ingested files reside in isolated volume storage with strict extension validation (`.pdf`, `.md`, `.txt`, `.py`, `.json`).
