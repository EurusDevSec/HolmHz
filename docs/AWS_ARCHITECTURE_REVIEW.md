# 📊 Đánh giá Kiến trúc AWS — HolmHz Deployment
> **Review date**: 2026-04-16 | **Reviewer**: Antigravity AI

---

## 🟢 ĐIỂM MẠNH — Bạn đã làm đúng

| # | Điểm tốt | Tại sao đúng |
|---|---------|-------------|
| ✅ | **Serverless (Lambda)** | Traffic thấp (research demo) → pay-per-request, không tốn tiền idle |
| ✅ | **Lambda Container Image từ ECR** | ONNX Runtime ~100MB+ → quá lớn cho Lambda Layer (250MB limit unzipped), Container Image Lambda cho phép tới 10GB |
| ✅ | **CloudFront** | CDN giảm latency, cache static responses, shield cho origin |
| ✅ | **API Gateway** | Proper HTTP entrypoint, có throttling built-in |
| ✅ | **S3 lưu Heatmap** | Đúng — không trả binary ảnh trong response JSON, thay bằng URL |
| ✅ | **CloudWatch** | Monitoring/logging cơ bản |
| ✅ | **GitHub Actions CI/CD** | Automate build → push ECR → update Lambda |
| ✅ | **Terraform (IaC)** | Infrastructure as Code — reproducible, version controlled |
| ✅ | **Docker image** | Containerize đảm bảo consistent environment |

---

## 🔴 VẤN ĐỀ CỦA KIẾN TRÚC

### ❌ 1. Luồng dữ liệu (Data Flow) chưa rõ ràng

**Vấn đề**: Nhìn diagram không thấy được request đi theo đường nào, response trả về đâu.

```
❓ Hiện tại không rõ:
User → CloudFront → API Gateway → Lambda → ???
Lambda → S3 (save heatmap) → ??? user nhận kết quả kiểu gì?

✅ Đúng phải là 1 trong 2:
Option A (Synchronous):
User → CloudFront → APIGW → Lambda → S3 (save PNG)
Lambda → return JSON {label, prob, heatmap_url: "https://cdn.../heatmap.png"}
User → CloudFront/S3 → download heatmap PNG

Option B (Async):
User → POST ảnh → APIGW → Lambda → SQS → Lambda Inference
→ S3 (save heatmap)
→ SNS/WebSocket notify user
User → GET result
```

**Câu hỏi**: Bạn chọn sync hay async? Inference ~1.5s → sync đủ ổn, không cần async.

---

### ❌ 2. Thiếu WAF (Web Application Firewall)

**Vấn đề**: CloudFront không có WAF → exposed to:
- DDoS attacks
- Người upload ảnh 100MB phá hệ thống
- Excessive requests (bill shock)

```
❌ Hiện tại:
User → CloudFront → API Gateway

✅ Nên thêm:
User → CloudFront + WAF → API Gateway
               ↓
         Rules: Rate limit (100 req/IP/min)
                Max file size (5MB)
                Block suspicious patterns
```

---

### ❌ 3. Không có Authentication / Authorization

**Vấn đề**: API Gateway hiện tại public hoàn toàn → bất kỳ ai cũng gọi được → Lambda runs → bạn tốn tiền.

```
❌ Hiện tại: API Gateway open

✅ Options (chọn 1):
- API Key đơn giản (cho research demo)
- AWS Cognito User Pool (nếu cần user login)
- IAM Auth (cho internal/machine-to-machine)
- Lambda Authorizer (custom JWT validation)
```

Với research demo: **API Key** là đơn giản nhất.

---

### ❌ 4. IAM Roles chưa được thiết kế (Least Privilege)

**Vấn đề**: Diagram không thể hiện rõ IAM role nào được gán cho ai.

```
✅ Cần định nghĩa rõ:

Lambda Execution Role:
  - s3:PutObject → bucket/heatmaps/* (ONLY)
  - s3:GetObject → bucket/model/efficientnet_b0.onnx (ONLY)
  - logs:CreateLogGroup, logs:PutLogEvents (CloudWatch)
  - ecr:GetDownloadUrlForLayer, ecr:BatchGetImage
  KHÔNG cần: s3:DeleteObject, s3:ListAllMyBuckets, ...

GitHub Actions OIDC Role:
  - ecr:GetAuthorizationToken
  - ecr:PutImage, ecr:InitiateLayerUpload
  - lambda:UpdateFunctionCode
  KHÔNG cần: lambda:DeleteFunction, s3:*, ...
```

**Gợi ý**: Dùng **OIDC (OpenID Connect)** cho GitHub Actions thay vì Long-lived Access Keys — best practice 2025.

---

### ❌ 5. S3 → CloudWatch "Trigger Event" — mục đích không rõ

**Vấn đề**: Mũi tên từ S3 lên CloudWatch với label "Trigger Event" — không rõ đây là gì.

```
❓ Bạn đang muốn làm gì?

Option A: S3 Event → CloudWatch Logs (audit trail)
→ Dùng S3 Server Access Logging → CloudWatch

Option B: S3 PutObject Event → Trigger Lambda khác
→ Dùng S3 Event Notification → Lambda
→ Không cần qua CloudWatch

Option C: CloudWatch alarm khi S3 bucket size > threshold
→ CloudWatch Metric + Alarm → SNS notification
```

Hãy làm rõ intent trước khi vẽ mũi tên.

---

### ❌ 6. Không có VPC (có thể chấp nhận được, nhưng cần giải thích)

**Vấn đề**: Lambda không nằm trong VPC.

```
✅ Với kiến trúc này: Lambda không cần VPC nếu:
- Không cần kết nối RDS/ElastiCache trong private subnet
- Chỉ cần gọi S3 (dùng VPC Endpoint nếu muốn private)
- Không có database private

⚠️ Nếu sau này thêm database (RDS/Aurora) → bắt buộc VPC
   → Plan trước để không phải refactor
```

**Recommendation**: Đặt Lambda trong VPC ngay từ đầu (với private subnet + VPC Endpoint for S3 và ECR) — dễ mở rộng sau.

---

### ❌ 7. Lambda Cold Start chưa được xử lý

**Vấn đề**: Lambda Container Image với ONNX Runtime:
- First invocation: load container → load model → inference = **5–15 giây cold start**
- Sau đó: ~1.5 giây/inference (warm)

```
✅ Options:
1. Provisioned Concurrency (keep N instances warm — có cost)
2. Scheduled EventBridge Ping mỗi 5 phút (free, "poor man's warming")
3. Lambda SnapStart (Java only, không áp dụng Python)
4. Pre-load model khi container init (outside handler function)
```

Đã làm chưa? Pre-load model là bắt buộc:
```python
# ✅ ĐÚNG: Load model NGOÀI handler (chạy 1 lần khi container start)
model = ort.InferenceSession("model.onnx")

def handler(event, context):
    result = model.run(...)  # Reuse model đã load
    return result

# ❌ SAI: Load model TRONG handler (load mỗi request = chậm)
def handler(event, context):
    model = ort.InferenceSession("model.onnx")  # ← mỗi request tải lại!
```

---

### ❌ 8. S3 Bucket chưa được phân tách

**Vấn đề**: 1 S3 bucket lưu tất cả (model file + user heatmaps) → bad practice.

```
❌ Hiện tại (ẩn ý): 1 bucket for everything

✅ Nên tách:
holmhz-models-bucket/          (private, versioned)
  └── efficientnet_b0.onnx

holmhz-results-bucket/         (pre-signed URL access)
  └── heatmaps/
      └── {uuid}/heatmap.png   (lifecycle: xóa sau 24h)
```

**Lý do**: Least privilege — Lambda chỉ cần write vào results bucket, không cần access models bucket sau khi init (hoặc dùng Lambda Layer/container để bundle model).

---

### ⚠️ 9. Secrets/Config Management

**Vấn đề**: Không thấy nơi lưu secrets/config.

```
✅ Dùng:
- AWS Systems Manager Parameter Store (free tier): non-sensitive config
- AWS Secrets Manager (có cost): sensitive keys

Ví dụ cho HolmHz:
  /holmhz/model-bucket-name   → Parameter Store
  /holmhz/api-key-salt        → Secrets Manager
```

**Không bao giờ** hardcode trong code hoặc Dockerfile.

---

### ⚠️ 10. CI/CD Pipeline chưa đủ stages

**Vấn đề**: GitHub Actions → Docker → ECR → Lambda — đủ không?

```
✅ Pipeline production-grade nên có:
Dev push code
  → GitHub Actions trigger
  → Run tests (pytest)
  → Build Docker image
  → Push to ECR (tag: git SHA)
  → Update Lambda (blue/green hoặc alias)
  → Smoke test (invoke Lambda test event)
  → Done ✅

❌ Thiếu: automated tests before deploy!
   Nếu push broken code → Lambda broken → user bị lỗi
```

---

## 📋 TỔNG KẾT ĐÁNH GIÁ

| Pillar | Điểm | Nhận xét |
|--------|------|---------|
| **Operational Excellence** | 7/10 | CI/CD OK, thiếu automated test stage, CloudWatch chưa có alarms |
| **Security** | 4/10 | Thiếu WAF, Auth, IAM Roles chưa được định nghĩa, no Secrets Manager |
| **Reliability** | 7/10 | Lambda+APIGW tự HA, thiếu xử lý cold start |
| **Cost Optimization** | 8/10 | Serverless đúng hướng, S3 lifecycle chưa có |
| **Performance** | 6/10 | Cold start chưa xử lý, model pre-load chưa thấy |
| **Tổng** | **6.4/10** | Kiến trúc tốt về big picture, cần bổ sung security layer |

---

## 🎯 ƯU TIÊN SỬA NGAY (TOP 3)

```
1. 🔴 Thêm WAF vào CloudFront
   → Tránh bill shock, DDoS
   → 30 phút implement với Terraform

2. 🔴 Thêm API Gateway Authentication (API Key minimum)
   → Không để public API
   → Terraform aws_api_gateway_api_key

3. 🟡 Làm rõ Data Flow trong diagram
   → Vẽ lại với mũi tên có label rõ ràng
   → Ghi rõ: sync hay async? Pre-signed URL hay inline response?
```

---

## ✅ ĐIỂM XUẤT SẮC CỦA BẠN

> **Terraform + GitHub Actions + Lambda Container + ECR** là stack rất chuẩn,  
> hiện đại, và phù hợp với workload này.  
> Rất nhiều junior dev không nghĩ tới IaC ngay từ đầu — bạn đã làm đúng! 🎉

**Kết luận**: Kiến trúc big picture đúng hướng. Bổ sung security layer (WAF + Auth + IAM roles explicit) là việc cần làm tiếp theo.
