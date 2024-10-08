CREATE TABLE customer_churn (
    customer_id INT AUTO_INCREMENT PRIMARY KEY,
    gender VARCHAR(10),
    age INT,
    tenure INT,
    balance DECIMAL(15, 2),
    products_number INT,
    credit_score INT,
    is_active_member VARCHAR(3),
    estimated_salary DECIMAL(15, 2),
    churn INT
);

INSERT INTO customer_churn (gender, age, tenure, balance, products_number, credit_score, is_active_member, estimated_salary, churn) VALUES
('Male', 42, 3, 0.00, 1, 619, 'Yes', 101348.88, 1),
('Female', 41, 1, 83807.86, 1, 608, 'No', 112542.58, 0),
('Male', 42, 8, 159660.80, 3, 502, 'Yes', 113931.57, 1),
('Female', 39, 1, 0.00, 2, 476, 'Yes', 102564.59, 0),
('Female', 43, 2, 125510.82, 1, 500, 'Yes', 125510.82, 1),
('Male', 44, 8, 113755.78, 2, 717, 'Yes', 113755.78, 0),
('Female', 50, 7, 0.00, 2, 495, 'No', 14406.41, 1),
('Male', 29, 4, 0.00, 2, 576, 'Yes', 158684.81, 0),
('Female', 36, 10, 113755.78, 3, 458, 'Yes', 113755.78, 1),
('Male', 48, 5, 0.00, 1, 712, 'Yes', 120026.32, 0),
('Female', 37, 6, 0.00, 1, 618, 'No', 118742.26, 0),
('Male', 45, 3, 0.00, 3, 697, 'Yes', 149756.71, 1),
('Female', 35, 10, 0.00, 3, 554, 'No', 150040.60, 0),
('Male', 58, 7, 0.00, 1, 632, 'Yes', 134603.88, 1),
('Female', 24, 9, 0.00, 2, 640, 'Yes', 138183.46, 0),
('Female', 45, 2, 0.00, 1, 570, 'No', 154360.36, 0),
('Male', 50, 1, 0.00, 2, 667, 'No', 165657.21, 1),
('Female', 53, 1, 0.00, 2, 606, 'Yes', 148431.42, 0),
('Male', 36, 1, 0.00, 2, 692, 'Yes', 118569.58, 1),
('Female', 40, 4, 0.00, 2, 582, 'Yes', 100383.94, 0),
('Male', 42, 6, 0.00, 1, 590, 'No', 144745.18, 0),
('Female', 29, 5, 0.00, 2, 622, 'Yes', 142051.07, 0),
('Male', 46, 3, 0.00, 2, 623, 'Yes', 134603.88, 1),
('Female', 50, 1, 0.00, 2, 516, 'No', 147995.61, 0),
('Male', 56, 9, 0.00, 1, 736, 'Yes', 133558.39, 0),
('Female', 42, 2, 0.00, 2, 557, 'No', 170732.81, 0),
('Male', 35, 7, 0.00, 1, 610, 'No', 151819.85, 1),
('Female', 40, 3, 0.00, 1, 693, 'Yes', 149756.71, 0),
('Male', 35, 3, 0.00, 2, 535, 'Yes', 141533.19, 1),
('Female', 35, 2, 0.00, 1, 574, 'No', 157264.32, 0),
('Male', 42, 7, 0.00, 1, 581, 'Yes', 136815.64, 0),
('Female', 30, 4, 0.00, 2, 578, 'No', 138293.63, 0),
('Male', 49, 4, 0.00, 1, 650, 'Yes', 125937.07, 1),
('Female', 53, 1, 0.00, 1, 558, 'No', 157249.43, 0),
('Male', 55, 4, 0.00, 1, 700, 'Yes', 121516.79, 0),
('Female', 56, 9, 0.00, 1, 713, 'Yes', 138953.45, 0),
('Male', 36, 3, 0.00, 1, 726, 'No', 109376.78, 0),
('Female', 33, 7, 0.00, 1, 594, 'Yes', 137215.22, 0),
('Male', 50, 7, 0.00, 2, 585, 'Yes', 137337.20, 1),
('Female', 54, 3, 0.00, 2, 708, 'Yes', 118026.12, 0),
('Male', 47, 8, 0.00, 2, 594, 'No', 156104.85, 1);
