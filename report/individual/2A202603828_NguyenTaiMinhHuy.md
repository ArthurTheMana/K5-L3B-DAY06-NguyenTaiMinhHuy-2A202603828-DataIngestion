# Báo cáo cá nhân — Day 6

Họ và tên: Nguyễn Tài Minh Huy

MSSV: 2A202603828

## 1. Dữ liệu và pipeline

### 1.1. Chủ đề và tài liệu

Trong bài thực hành này, mình xây dựng một hệ thống có thể đọc tài liệu PDF về AI và Machine Learning, chuyển nội dung thành vector và tìm những đoạn văn có liên quan đến câu hỏi. Mục tiêu của mình không phải là tạo một chatbot hoàn chỉnh mà là kiểm tra cách chuẩn bị dữ liệu ảnh hưởng đến kết quả tìm kiếm như thế nào.

Mình sử dụng hai tài liệu. Tài liệu thứ nhất là *A Gentle Introduction to Artificial Intelligence and Machine Learning*, được lưu với mã `uow_ai_ml`, có 56 trang PDF. Tài liệu thứ hai là *A non-technical introduction to machine learning* của Olivier Colliot, được lưu với mã `colliot_ml_2023`, có 26 trang PDF. Nguồn của tài liệu thứ hai được ghi trong manifest tại [HAL](https://hal.science/hal-03957125v3).

Tổng cộng, bộ dữ liệu có 82 trang PDF. Kết quả kiểm tra trong `retrieval_metrics.json` ghi nhận 44 trang nội dung hữu ích của tài liệu UOW và 23 trang của tài liệu Colliot, tương đương 67 trang. Việc đếm sử dụng số trang PDF thực tế, bắt đầu từ trang 1 và loại các trang không có nội dung đủ hữu ích như trang bìa hoặc trang chỉ chứa thông tin phụ.

Tuy nhiên, mình nhận thấy bản `corpus_manifest.json` hiện được lưu vẫn ghi `valid_content_pages = 0` ở cả hai tài liệu, chưa khớp với kết quả 67 trang trong metrics. Đây là thông tin mình cần cập nhật và kiểm tra lại trước khi nộp bài, thay vì coi manifest hiện tại là đã hoàn chỉnh.

Về quyền sử dụng, manifest ghi tài liệu Colliot có giấy phép `CC BY 4.0`, còn tài liệu UOW đang ở trạng thái `TO_VERIFY`. Vì vậy, mình cần xác minh quyền chia sẻ tài liệu UOW trước khi đưa file PDF hoặc nội dung trích xuất của nó lên repository công khai.

### 1.2. Xử lý và chuẩn hóa dữ liệu

Ban đầu, mình gặp vấn đề khi trích xuất PDF bằng `pypdf` vì một số từ bị tách thành các ký tự rời nhau. Sau đó, mình chuyển sang `PyMuPDF` để lấy văn bản theo từng trang. Mình chuẩn hóa Unicode, khoảng trắng và ký tự xuống dòng, đồng thời giữ lại số trang và thông tin nguồn để có thể kiểm tra kết quả sau này.

Pipeline của mình thực hiện theo thứ tự: đọc PDF, chuẩn hóa văn bản, chia chunk, tạo embedding, lưu vector vào cơ sở dữ liệu, truy xuất Top 3 và đánh giá kết quả. Một hạn chế vẫn còn là văn bản lấy từ hình, công thức hoặc bố cục nhiều cột có thể khó đọc. Mình chưa triển khai OCR để khôi phục hoàn toàn những trang không trích xuất được.

### 1.3. Embedding, cơ sở dữ liệu và cách chạy

Mình dùng mô hình `sentence-transformers/all-MiniLM-L6-v2` thông qua thư viện Sentence Transformers chạy trên máy local bằng CPU. Mỗi đoạn văn và câu hỏi được chuyển thành vector 384 chiều. Mình giữ nguyên mô hình này cho cả ba chiến lược, để việc so sánh không bị ảnh hưởng bởi thay đổi embedding.

Mình sử dụng PostgreSQL 16 với pgvector 0.8.7, chạy bằng Docker. Dữ liệu được lưu trong bảng `day6_chunks`, có văn bản, vector và metadata. Khi tìm kiếm, chương trình tính cosine distance, sắp xếp kết quả và lấy ba chunks gần nhất. Với lượng dữ liệu nhỏ của bài thực hành, chương trình dùng truy vấn tìm kiếm chính xác thay vì cấu hình chỉ mục tìm kiếm gần đúng.

Mình chạy chương trình từ thư mục `starter` bằng lệnh `python -m src.run`. Để chuẩn bị cơ sở dữ liệu, mình sử dụng `docker compose -p day6_student_starter up -d --wait`. Lần chạy đã ghi nhận 14,09 giây cho quá trình tạo chunks, embedding và lưu vector của A và B. Đây là thời gian tổng của bước xử lý đó, không phải thời gian riêng từng chiến lược hoặc thời gian trả lời một câu hỏi. Chi phí API là 0 USD vì mình chạy embedding tại máy, nhưng mình chưa đo chi phí điện năng hoặc tài nguyên máy.

### 1.4. Metadata

Mỗi chunk được gắn `document_id`, `source`, `page` và `document_type`. Hai giá trị của `document_type` là `lecture_slides` cho tài liệu UOW và `book_chapter` cho tài liệu Colliot. Các giá trị này được xác định dựa trên loại tài liệu, không phải dự đoán từ câu hỏi.

Mình chọn `document_type` vì nó giúp giới hạn phạm vi tìm kiếm khi câu hỏi yêu cầu thông tin từ slide bài giảng hoặc chương sách. Các trường `source` và `page` giúp mình truy ngược về đúng PDF và trang nguồn để kiểm tra bằng chứng.

## 2. Cấu hình và benchmark

### 2.1. Ba chiến lược

**Strategy A: Fixed Size Chunking.** Mình chia văn bản thành các đoạn tối đa 500 ký tự và dùng overlap 50 ký tự. Cách này đơn giản, dễ kiểm soát, nhưng có thể cắt ngang một câu hoặc một ý đang được giải thích. Kết quả tạo ra 190 chunks.

**Strategy B: Paragraph Based Chunking.** Mình ưu tiên giữ ranh giới đoạn văn và giới hạn kích thước mỗi chunk ở 500 ký tự, không dùng overlap. Nếu một đoạn quá dài, chương trình vẫn phải cắt nhỏ. Cách này tạo ra 181 chunks và có khả năng giữ ngữ cảnh tốt hơn ở những chỗ văn bản có cấu trúc rõ ràng.

**Strategy C: Paragraph Based Chunking kết hợp Metadata Filtering.** Mình sử dụng lại đúng 181 chunks và embeddings của B, sau đó bổ sung bộ lọc `document_type` khi câu hỏi có yêu cầu cụ thể. C không phải một thuật toán chunking thứ ba. Điểm khác nhau duy nhất giữa B và C là việc áp dụng bộ lọc lúc truy xuất.

### 2.2. Bộ câu hỏi và cách đánh giá

Mình chuẩn bị 20 câu hỏi có đáp án tham chiếu, nguồn, số trang và đoạn bằng chứng. Bộ câu hỏi gồm 15 câu Dev và 5 câu Test, được dùng giống nhau cho cả A, B và C. Kết quả kiểm tra trong metrics cho biết cả 20 đoạn evidence tham chiếu đều có mặt ở trang nguồn đã ghi.

Có bốn câu chứa yêu cầu lọc theo loại tài liệu là Q05, Q10, Q12 và Q20. Trong đó Q10 và Q20 thuộc tập Test. Khi chạy Strategy C, chương trình chỉ dùng filter nếu câu hỏi cung cấp filter trước khi truy xuất.

Mình sử dụng hai chỉ số chính. `Hit@1` cho biết có bao nhiêu câu mà chunk đứng đầu chứa đúng bằng chứng được yêu cầu. `Evidence Hit@3` cho biết có bao nhiêu câu tìm được bằng chứng trong ba chunks đầu tiên. Bộ chấm tự động hiện dựa trên việc đối chiếu đoạn evidence với nguồn và trang tham chiếu. Vì vậy, những đoạn có ý nghĩa tương đương nhưng không trùng hoàn toàn văn bản vẫn cần được đọc lại thủ công.

Về tiêu chí lựa chọn, mình ưu tiên `Evidence Hit@3` trên Dev, sau đó xem xét Grounding, khả năng lọc metadata và tài nguyên sử dụng. Tuy nhiên, mình chưa có bản ghi độc lập chứng minh thứ tự ưu tiên này đã được chốt trước khi xem Test, nên mình trình bày đây là tiêu chí phân tích khi tổng kết chứ không nhận là một kế hoạch đã đăng ký trước.

## 3. Kết quả

### 3.1. Retrieval Accuracy

| Strategy | Split | Hit@1 | Evidence Hit@3 | Số chunks | Thời gian | Chi phí API |
| :--- | :--- | :--- | :--- | ---: | :--- | :--- |
| A | Dev | 6/15, 40% | 10/15, 66,67% | 190 | Không đo riêng | 0 USD |
| A | Test | 2/5, 40% | 3/5, 60% | 190 | Không đo riêng | 0 USD |
| B | Dev | 6/15, 40% | 12/15, 80% | 181 | Không đo riêng | 0 USD |
| B | Test | 2/5, 40% | 3/5, 60% | 181 | Không đo riêng | 0 USD |
| C | Dev | 6/15, 40% | 12/15, 80% | 181 dùng lại B | Không đo riêng | 0 USD |
| C | Test | 2/5, 40% | 3/5, 60% | 181 dùng lại B | Không đo riêng | 0 USD |

Kết quả cho thấy B và C tìm được nhiều bằng chứng hơn A trên tập Dev, với 12/15 câu so với 10/15 câu. Tuy nhiên, cả ba chiến lược đều đạt 3/5 câu trên Test. Chỉ số Hit@1 cũng không thay đổi giữa ba chiến lược. Vì vậy, mình chưa thể nói rằng một chiến lược luôn chính xác hơn những chiến lược còn lại.

### 3.2. Coherence, Metadata và Grounding

| Strategy | Coherence trung bình /2 | Metadata đủ và đúng | Tuân thủ filter | Grounding trên 5 câu Test |
| :--- | :--- | :--- | :--- | :--- |
| A | 1,10, n = 10 | 15/15 | N/A | 0 điểm: 0 câu; 1 điểm: 2 câu; 2 điểm: 3 câu; trung bình 1,60/2 |
| B | 1,10, n = 10 | 15/15 | N/A | 0 điểm: 0 câu; 1 điểm: 1 câu; 2 điểm: 4 câu; trung bình 1,80/2 |
| C | 1,10, n = 10 | 15/15 | 2/2 câu Test, 4/4 câu toàn bộ benchmark | 0 điểm: 0 câu; 1 điểm: 1 câu; 2 điểm: 4 câu; trung bình 1,80/2 |

Phần Coherence được chấm theo thang 0, 1 và 2 trên 10 vị trí nguồn cho mỗi chiến lược. Trong mỗi nhóm có một chunk được chấm 0, bảy chunks được chấm 1 và hai chunks được chấm 2. Điểm trung bình là 1,10/2. Mình nhận thấy 10 chunks được chọn của A có nội dung giống hệt 10 chunks tương ứng trong B và C. Vì thế, mẫu này chưa đủ để kết luận cách chia đoạn của B có Coherence tốt hơn A. Nếu có thêm thời gian, mình sẽ chọn thêm các vị trí mà A và B thực sự tạo ra chunks khác nhau.

Về Metadata, file audit hiện ghi 45/45 kết quả Top 3 có đầy đủ và đúng các trường nguồn, trang và loại tài liệu khi đối chiếu với PDF. Trong đó mỗi chiến lược có 15/15 bản ghi. Tuy nhiên, cột `reviewer_confirm_y_n` trong file audit vẫn để trống. Mình cần kiểm tra và xác nhận lại từng dòng trước khi xem đây là kết quả đánh giá thủ công cuối cùng. Với Strategy C, cả bốn câu hỏi có filter đều trả về Top 3 tuân thủ đúng loại tài liệu, bao gồm hai câu Test Q10 và Q20.

Phần Grounding được chấm trên năm câu Test Q09, Q10, Q18, Q19 và Q20. Các điểm trong bảng là nhãn đánh giá đề xuất đã điền vào CSV để mình rà soát. A có trung bình 1,60/2, còn B và C có trung bình 1,80/2. Grounding ở đây chỉ đánh giá bằng chứng truy xuất có đủ để trả lời hay không, chứ không phải độ chính xác của câu trả lời từ LLM.

Mình lưu kết quả truy xuất trong `starter/results/retrieval_traces.json`, số liệu tổng hợp trong `starter/results/retrieval_metrics.json`, điểm Coherence trong `starter/results/manual_chunk_coherence.csv`, điểm Grounding trong `starter/results/manual_test_grounding.csv` và phần rà soát metadata trong `starter/results/metadata_audit.csv`. Cấu hình thí nghiệm được ghi trong `artifacts/experiment_log.csv`.

## 4. Phân tích và kết luận

### 4.1. So sánh A với B và B với C

Khi so sánh A với B, mình thấy việc ưu tiên đoạn văn giúp tăng Evidence Hit@3 trên tập Dev từ 66,67% lên 80%. Đồng thời, số chunks giảm từ 190 xuống 181. Tuy nhiên, kết quả Test không thay đổi nên đây mới là tín hiệu tốt trong bộ dữ liệu mình sử dụng, chưa phải bằng chứng rằng Paragraph Based Chunking luôn tốt hơn Fixed Size Chunking.

Khi so sánh B với C, cả hai có kết quả Hit@1, Evidence Hit@3 và Grounding giống nhau. Lợi ích của C nằm ở khả năng giới hạn kết quả theo loại tài liệu. Ví dụ, với Q05 hỏi về Supervised Learning trong slide UOW, A không tìm được đúng bằng chứng trong Top 3, B tìm được ở vị trí thứ ba, còn C tìm được ở vị trí thứ hai. Mình thấy filter có ích khi người dùng muốn tìm đúng tài liệu, nhưng kết quả này không đủ để kết luận C luôn cải thiện độ chính xác.

### 4.2. Hai trường hợp tìm kiếm chưa tốt

**Q18: Feature Engineering.** Câu hỏi yêu cầu giải thích Feature Engineering được dùng để làm gì trong chương sách Colliot. Cả ba chiến lược đều có Evidence Hit@3 tự động bằng 0 vì không tìm được toàn bộ đoạn trích tham chiếu theo đúng tiêu chí đối chiếu. Tuy nhiên, khi đọc Top 3, mình thấy A có thông tin liên quan nhưng chưa đủ phần giải thích về biểu diễn số, nên đề xuất chấm Grounding 1. Với B và C, đoạn ở trang 20 đề cập rõ hơn đến việc tạo các đặc trưng thành vector đầu vào, nên mình đề xuất điểm 2. Trường hợp này cho thấy kết quả tự động có thể bỏ sót bằng chứng có ý nghĩa tương đương.

**Q19: Linear Regression.** Câu hỏi hỏi điều kiện để chọn mô hình tốt nhất. Kết quả tìm kiếm có đề cập đến hàm chi phí, nhưng phần câu giải thích tiêu chí tối thiểu hóa bị cắt ở ranh giới chunk. Vì vậy, mình đề xuất Grounding 1 cho cả ba chiến lược. Đây là ví dụ cho thấy một chunk xuất hiện đúng tài liệu và trang vẫn có thể chưa đủ thông tin để trả lời.

### 4.3. Chiến lược mình lựa chọn

Nếu phải chọn một cấu hình làm hướng phát triển tiếp theo, mình nghiêng về **Strategy C**. Lý do là C giữ được kết quả truy xuất của B trong benchmark này và có thêm khả năng lọc theo loại tài liệu mà không cần tạo một bộ embeddings riêng. Với những câu hỏi không yêu cầu filter, C hoạt động giống B. Với những câu yêu cầu chỉ xem slide hoặc chương sách, C có thể thu hẹp phạm vi tìm kiếm theo đúng ngữ cảnh.

Dù vậy, mình không xem C là chiến lược có độ chính xác cao nhất trong mọi trường hợp. Dữ liệu Test chỉ có năm câu và kết quả ba chiến lược bằng nhau. Việc lọc quá chặt còn có thể loại bỏ bằng chứng hữu ích nếu metadata của nguồn không đúng hoặc câu hỏi được gắn filter chưa phù hợp.

### 4.4. Hạn chế và hướng cải tiến

Bài làm của mình còn một số hạn chế. Thứ nhất, tập Test nhỏ nên mỗi câu hỏi có ảnh hưởng khá lớn đến tỷ lệ phần trăm. Thứ hai, việc đối chiếu exact evidence quote có thể chấm thiếu những đoạn có ý nghĩa tương đương. Thứ ba, mẫu Coherence hiện tại chưa làm nổi bật khác biệt giữa A và B vì các đoạn được chọn giống nhau. Thứ tư, dữ liệu trích xuất từ PDF có thể bị lỗi ở hình minh họa và công thức. Cuối cùng, mình mới đo thời gian xử lý chung, chưa đo riêng tốc độ truy vấn, mức sử dụng CPU, RAM hoặc điện năng.

Nếu thực hiện lại, mình sẽ hoàn thiện manifest để số trang khớp với kết quả đã kiểm tra, bổ sung cách ghi nhận lỗi PDF, chọn mẫu Coherence có khả năng phân biệt hai thuật toán chunking, thử một vài giá trị chunk size khác nhau trên Dev và đo thời gian truy vấn riêng cho từng cấu hình. Mình cũng muốn thử reranking hoặc kết hợp tìm kiếm theo từ khóa, nhưng chỉ sau khi có một bộ Test mới hoặc một tập đánh giá độc lập để tránh điều chỉnh dựa trên kết quả đã xem.

### 4.5. Công cụ hỗ trợ và tự đánh giá

Trong quá trình làm bài, mình có sử dụng ChatGPT để hỗ trợ hiểu yêu cầu, giải thích code, kiểm tra kết quả và gợi ý cách chấm thủ công. Mình vẫn cần tự đối chiếu các điểm đánh giá với nội dung PDF và giải thích được vì sao mình chọn từng chiến lược. Những nhận xét và nhãn chấm còn ở trạng thái đề xuất không nên được trình bày như kết quả xác nhận độc lập.

Mình hiểu rõ hơn rằng chất lượng tìm kiếm không chỉ phụ thuộc vào embedding model. Cách trích xuất văn bản, chia chunks và sử dụng metadata đều có thể ảnh hưởng đến bằng chứng mà hệ thống lấy được. Với corpus hiện tại, mình chọn C làm hướng phát triển tiếp theo vì khả năng kiểm soát phạm vi tài liệu và đồng thời vẫn cần thêm đánh giá trước khi khẳng định hiệu quả lâu dài.