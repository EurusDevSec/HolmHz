# 🏗️ AWS Architecture Design — HolmHz Deepfake Detector
> **Mục đích**: Gợi ý để bạn tự suy nghĩ & vẽ kiến trúc deploy hệ thống HolmHz  
> **Không phải đáp án** — đây là câu hỏi dẫn dắt theo 5 pillars AWS Well-Architected

---

## 📌 Bối cảnh hệ thống cần deploy

Trước khi vẽ, hãy hiểu rõ workload:

| Thành phần | Mô tả công nghệ |
|-----------|----------------|
| **ML Model** | EfficientNet-B0, file `.onnx` (~20MB) |
| **Inference** | Nhận ảnh → trả về `P(Fake)` + Grad-CAM heatmap |
| **Input** | Ảnh JPEG/PNG, tối đa vài MB |
| **Output** | JSON `{label, prob, heatmap_url}` |
| **Traffic** | Thấp (research demo), không phải production scale |
| **State** | Stateless (mỗi request độc lập) |

---

## 🧱 PILLAR 1 — Operational Excellence
*"Run and monitor systems to deliver business value"*

### Câu hỏi gợi ý:

**Deployment:**
- [ ] Bạn sẽ dùng cách nào để đóng gói model + code? → Docker? ZIP?
- [ ] CI/CD pipeline của bạn trông như thế nào? GitHub Actions → deploy đâu?
- [ ] Khi update model mới (v10, v11), làm sao deploy **không downtime**?

**Observability:**
- [ ] Làm sao biết inference đang chậm hay có lỗi? → Log ở đâu? Metric nào?
- [ ] AWS có service nào cho **centralized logging**?
- [ ] Dùng service nào để **alert** khi có error rate tăng?

> 💡 Hint: Nghĩ đến **CloudWatch**, **X-Ray** cho distributed tracing

---

## 🔒 PILLAR 2 — Security (Least Privilege)
*"Protect information, systems, and assets"*

### Câu hỏi gợi ý:

**Identity & Access:**
- [ ] Model file `.onnx` lưu trên S3 — **IAM Role** của Lambda/EC2 cần policy gì? Chỉ `s3:GetObject` hay `s3:*`?
- [ ] API của bạn có cần authentication không (API Key, Cognito, JWT)?
- [ ] Nếu dùng nhiều service (Lambda, S3, ECR) — mỗi service cần **role riêng** hay dùng chung?

**Network:**
- [ ] API có nên public internet hoàn toàn không?
- [ ] Bạn có cần đặt trong **VPC** không? Khi nào thì cần?
- [ ] **Security Group** của bạn cho phép port nào? Có nên restrict by IP không?

**Data:**
- [ ] Ảnh upload của user có cần **encrypt at rest** không?
- [ ] Ảnh có nên lưu lại sau khi inference không? → Tác động đến privacy policy?

> 💡 Hint: Nguyên tắc **Least Privilege** = chỉ cấp đúng permission cần thiết, không hơn

---

## 🔁 PILLAR 3 — Reliability (High Availability)
*"Recover from failures and meet demand"*

### Câu hỏi gợi ý:

**Single Point of Failure:**
- [ ] Nếu 1 instance EC2 chết → user có bị ảnh hưởng không?
- [ ] Để HA, bạn deploy ít nhất bao nhiêu **Availability Zones**?
- [ ] Load Balancer đặt ở đâu trong kiến trúc?

**Scaling:**
- [ ] Khi 100 user upload cùng lúc → hệ thống handle thế nào?
- [ ] **Auto Scaling Group** trigger theo metric nào? CPU? Request count?
- [ ] Inference model load vào memory mỗi request hay **cache lại**?

**Failure Recovery:**
- [ ] Nếu model file trên S3 bị xóa nhầm → có backup không?
- [ ] **S3 Versioning** có cần bật không?

> 💡 Hint: Nghĩ đến **ALB + Auto Scaling Group** hoặc **Lambda** (serverless tự scale)

---

## 💰 PILLAR 4 — Cost Optimization
*"Avoid unnecessary costs"*

### Câu hỏi gợi ý:

**Compute:**
- [ ] Traffic thấp (research demo) → EC2 `t3.medium` chạy 24/7 có hợp lý không?
- [ ] **Lambda** (pay-per-request) vs **EC2** (pay-per-hour) → cái nào rẻ hơn cho traffic thấp?
- [ ] Nếu dùng EC2, có nên dùng **Spot Instance** không? Trade-off là gì?

**Storage:**
- [ ] Ảnh upload của user sau khi inference → lưu mãi hay **tự động xóa** sau N ngày?
- [ ] **S3 Lifecycle Policy** có thể làm gì?
- [ ] Model file `.onnx` 20MB — có cần **S3 Intelligent-Tiering** không?

**Request Flow:**
- [ ] Nếu cùng 1 ảnh được upload nhiều lần → có nên **cache** kết quả không?
- [ ] **CloudFront** có giúp giảm cost không? Khi nào nên dùng CDN?

> 💡 Hint: Với workload thấp + intermittent → **Serverless** thường rẻ hơn EC2

---

## ⚡ PILLAR 5 — Performance Efficiency
*"Use computing resources efficiently"*

### Câu hỏi gợi ý:

**Latency:**
- [ ] Inference ONNX trên CPU: ~1.5s. Trên GPU: ~0.1s. GPU instance có worth không?
- [ ] Nếu dùng Lambda, **cold start** tác động thế nào đến latency?
- [ ] Model có thể load 1 lần và **reuse** giữa các invocations không?

**Caching:**
- [ ] **ElastiCache** (Redis) phù hợp cache kết quả inference không? Key là gì?
- [ ] Grad-CAM heatmap (ảnh) → lưu S3 và trả về pre-signed URL hay trả về base64?

**Region:**
- [ ] Deploy ở **region nào**? User ở Việt Nam → `ap-southeast-1` (Singapore)?
- [ ] Có cần **multi-region** không? Hay single region đủ?

---

## 🎨 GỢI Ý FLOW KIẾN TRÚC ĐỂ BẠN TỰ VẼ

Hãy tự vẽ diagram cho 2 phương án sau, rồi so sánh:

### Phương án A — Container-based
```
User
  ↓ HTTPS
[CloudFront]
  ↓
[ALB]
  ↓
[ECS Fargate / EC2 Auto Scaling Group]
  (FastAPI + ONNX Runtime)
  ↓           ↓
[S3]       [CloudWatch]
(model,    (logs, metrics)
 uploads)
```

**Câu hỏi khi vẽ A:**
- ECS vs EC2 — khi nào chọn cái nào?
- Fargate vs EC2 launch type — trade-off?
- IAM Role của ECS Task cần gì?

---

### Phương án B — Serverless
```
User
  ↓ HTTPS
[API Gateway]
  ↓
[Lambda]
  (Layer: ONNX Runtime)
  ↓               ↓
[S3]          [CloudWatch]
(model,        (logs)
 uploads)
```

**Câu hỏi khi vẽ B:**
- Lambda có RAM limit 10GB — ONNX model ~20MB + runtime → đủ không?
- Cold start của Lambda Python với ONNX Runtime mất bao lâu?
- Lambda Layer vs Container Image Lambda — khác nhau thế nào?
- Timeout limit của Lambda (15 phút) có đủ cho inference không?

---

## 📋 CHECKLIST TỰ ĐÁNH GIÁ TRƯỚC KHI FINALIZE

Khi vẽ xong, kiểm tra:

### Security
- [ ] Có IAM Role riêng cho từng service (không dùng root)?
- [ ] S3 bucket có `Block Public Access` = ON?
- [ ] API có rate limiting / throttling?
- [ ] Secrets (API keys) lưu ở **Secrets Manager / Parameter Store** hay hardcode?

### High Availability
- [ ] Deploy ≥ 2 AZ?
- [ ] Load balancer có health check?
- [ ] Auto Scaling có min=2?

### Cost
- [ ] Có đặt **Budget Alert** trên AWS Console chưa?
- [ ] S3 có Lifecycle Policy xóa temp uploads?
- [ ] CloudWatch logs có retention policy (không lưu mãi mãi)?

### Operational
- [ ] Có thể deploy version mới mà không downtime?
- [ ] Nếu lỗi xảy ra, có alarm notify về email/Slack không?

---

## 🔑 AWS SERVICES CẦN NGHIÊN CỨU

Đây là danh sách service liên quan — **bạn tự quyết định dùng cái nào**:

| Category | Services |
|----------|---------|
| **Compute** | EC2, ECS Fargate, Lambda, App Runner |
| **Networking** | VPC, ALB, CloudFront, API Gateway, Route 53 |
| **Storage** | S3, EFS |
| **Security** | IAM, Secrets Manager, WAF, Shield |
| **Monitoring** | CloudWatch, X-Ray |
| **CI/CD** | CodePipeline, CodeBuild, GitHub Actions |
| **Container** | ECR, ECS, Fargate |
| **Cache** | ElastiCache (Redis) |

---

## 📐 FORMAT GỢI Ý KHI VẼ

Khi bạn vẽ xong, hãy ghi lại:

```
1. Kiến trúc chọn: [A/B/C - hybrid]
2. Lý do chọn: ________________
3. Trade-offs chấp nhận: ________________
4. Estimated cost/month: $___
5. RTO (Recovery Time Objective): ___phút
6. RPO (Recovery Point Objective): ___phút
7. Điểm cải thiện trong tương lai: ________________
```

> 🎯 **Mục tiêu học tập**: Không có kiến trúc "đúng tuyệt đối" — mỗi lựa chọn có trade-off.  
> Quan trọng là bạn **hiểu tại sao** chọn, không phải chỉ vẽ cho xong.
