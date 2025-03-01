基於 **MySQL** 資料庫系統的實驗設計如下，這套實驗從基礎概念開始，逐步涵蓋關係代數、SQL 語法、鍵約束、查詢操作，最終形成完整的數據庫實驗體系。

---

# **實驗 1：MySQL 環境搭建與基本操作**
### **實驗目標**
1. 了解 MySQL 資料庫系統的基本概念。
2. 安裝並配置 MySQL，學習如何連接 MySQL 伺服器。
3. 熟悉 MySQL Workbench 或命令列工具，學習基本 SQL 操作。

### **實驗步驟**
1. **安裝 MySQL**
   - Windows 用戶：安裝 MySQL Community Server，使用 MySQL Workbench 或 `mysql` 命令行連接。
   - Linux 用戶：使用 `sudo apt install mysql-server` 安裝，然後 `mysql -u root -p` 進入 MySQL 命令行。

2. **建立 MySQL 資料庫**
   ```sql
   CREATE DATABASE SchoolDB;
   USE SchoolDB;
   ```

3. **建立 `Students` 表**
   ```sql
   CREATE TABLE Students (
       StudentID INT AUTO_INCREMENT PRIMARY KEY,
       Name VARCHAR(50) NOT NULL,
       Age INT CHECK (Age >= 18),
       Major VARCHAR(50)
   );
   ```

4. **插入數據**
   ```sql
   INSERT INTO Students (Name, Age, Major) 
   VALUES ('Alice', 21, 'Computer Science'),
          ('Bob', 22, 'Mathematics'),
          ('Charlie', 20, 'Physics');
   ```

5. **查詢數據**
   ```sql
   SELECT * FROM Students;
   ```

---

# **實驗 2：關係代數與 SQL 基本查詢**
### **實驗目標**
1. 學習 SQL 的基本查詢語句 `SELECT`，實現關係代數的選擇、投影和聯接。
2. 使用 `WHERE` 過濾數據，使用 `ORDER BY` 排序數據。

### **實驗步驟**
1. **選擇（Selection, σ）**：篩選年齡大於 20 的學生
   ```sql
   SELECT * FROM Students WHERE Age > 20;
   ```
2. **投影（Projection, π）**：顯示學生的姓名和專業
   ```sql
   SELECT Name, Major FROM Students;
   ```
3. **排序（Ordering）**：按年齡降序排序
   ```sql
   SELECT * FROM Students ORDER BY Age DESC;
   ```

---

# **實驗 3：關聯表與外鍵**
### **實驗目標**
1. 建立 `Courses` 表，並通過 `StudentID` 連接 `Students` 表，練習外鍵約束。
2. 執行表聯接查詢（JOIN）。

### **實驗步驟**
1. **建立 `Courses` 表**
   ```sql
   CREATE TABLE Courses (
       CourseID INT AUTO_INCREMENT PRIMARY KEY,
       CourseName VARCHAR(100) NOT NULL,
       StudentID INT,
       FOREIGN KEY (StudentID) REFERENCES Students(StudentID) ON DELETE CASCADE
   );
   ```

2. **插入數據**
   ```sql
   INSERT INTO Courses (CourseName, StudentID)
   VALUES ('Database Systems', 1),
          ('Linear Algebra', 2),
          ('Quantum Mechanics', 3);
   ```

3. **執行 INNER JOIN 查詢**
   ```sql
   SELECT Students.Name, Courses.CourseName
   FROM Students
   INNER JOIN Courses ON Students.StudentID = Courses.StudentID;
   ```

4. **測試外鍵約束**
   - 嘗試刪除 `Students` 表中的某一行：
     ```sql
     DELETE FROM Students WHERE StudentID = 1;
     ```
   - 如果 `ON DELETE CASCADE` 正確設置，對應的 `Courses` 表數據也會被刪除。

---

# **實驗 4：SQL 高級查詢**
### **實驗目標**
1. 學習 `GROUP BY`、`HAVING` 和聚合函數（`COUNT`、`SUM`、`AVG`）。
2. 掌握 `LIMIT` 和 `DISTINCT`。

### **實驗步驟**
1. **計算不同專業的學生人數**
   ```sql
   SELECT Major, COUNT(*) AS StudentCount 
   FROM Students 
   GROUP BY Major;
   ```

2. **顯示學生人數大於 1 的專業**
   ```sql
   SELECT Major, COUNT(*) AS StudentCount 
   FROM Students 
   GROUP BY Major 
   HAVING COUNT(*) > 1;
   ```

3. **使用 `LIMIT` 限制結果**
   ```sql
   SELECT * FROM Students LIMIT 2;
   ```

4. **使用 `DISTINCT` 查詢唯一的專業名稱**
   ```sql
   SELECT DISTINCT Major FROM Students;
   ```

---

# **實驗 5：數據庫事務與並發控制**
### **實驗目標**
1. 理解 MySQL 事務（Transaction）和 ACID 原則。
2. 學習如何使用 `COMMIT` 和 `ROLLBACK` 來管理事務。

### **實驗步驟**
1. **開始事務**
   ```sql
   START TRANSACTION;
   ```

2. **插入數據**
   ```sql
   INSERT INTO Students (Name, Age, Major) VALUES ('David', 23, 'Biology');
   ```

3. **回滾事務**
   ```sql
   ROLLBACK;
   ```

4. **確認數據是否仍然存在**
   ```sql
   SELECT * FROM Students;
   ```

5. **提交事務**
   ```sql
   START TRANSACTION;
   INSERT INTO Students (Name, Age, Major) VALUES ('Eve', 22, 'Chemistry');
   COMMIT;
   ```

---

# **實驗 6：索引與性能優化**
### **實驗目標**
1. 學習如何使用索引提高查詢性能。
2. 比較索引與非索引查詢的性能。

### **實驗步驟**
1. **創建索引**
   ```sql
   CREATE INDEX idx_major ON Students(Major);
   ```

2. **查詢分析**
   ```sql
   EXPLAIN SELECT * FROM Students WHERE Major = 'Computer Science';
   ```

3. **刪除索引**
   ```sql
   DROP INDEX idx_major ON Students;
   ```

---

# **實驗 7：視圖與存儲過程**
### **實驗目標**
1. 學習如何使用視圖簡化查詢。
2. 理解存儲過程（Stored Procedure）。

### **實驗步驟**
1. **創建視圖**
   ```sql
   CREATE VIEW StudentView AS
   SELECT Name, Major FROM Students;
   ```

2. **使用視圖**
   ```sql
   SELECT * FROM StudentView;
   ```

3. **創建存儲過程**
   ```sql
   DELIMITER //
   CREATE PROCEDURE GetAllStudents()
   BEGIN
       SELECT * FROM Students;
   END //
   DELIMITER ;
   ```

4. **調用存儲過程**
   ```sql
   CALL GetAllStudents();
   ```

---

這套實驗涵蓋了 MySQL 的基礎概念、關係代數、鍵約束、高級查詢、事務控制、索引優化和存儲過程，適合循序漸進地學習 **MySQL** 數據庫系統的基本原理與應用。
