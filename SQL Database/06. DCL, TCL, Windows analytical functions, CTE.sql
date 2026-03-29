-- DCL

create user "amittt"@"localhost" identified by "amit1" ;

grant select on class11.course to "amittt"@"localhost" ;

grant all on class11.course to "amittt"@"localhost" ;

grant all on class11.* to "amittt"@"localhost" ;

revoke select on class11.course from "amittt"@"localhost" ;

revoke all on class11.course from "amittt"@"localhost" ;

revoke all on class11.* from "amittt"@"localhost" ;

revoke all on class11.products from "amittt"@"localhost" ;

select * from products ;

select * from course ;

DROP USER 'amittt'@'localhost';

-- TCL

CREATE TABLE t_school(
ID INT, School_Name VARCHAR(40), 
Number_Of_Students INT, Number_Of_Teachers INT, Number_Of_Classrooms INT, EmailID VARCHAR(40));

INSERT INTO t_school(ID, School_Name, Number_Of_Students, Number_Of_Teachers,
 Number_Of_Classrooms, EmailID) 
 VALUES(1, "Boys Town Public School", 1000, 80, 12, "btps15@gmail.com"), 
 (2, "Guru Govind Singh Public School", 800, 35, 15, "ggps25@gmail.com"), 
 (3, "Delhi Public School", 1200, 30, 10, "dps101@gmail.com"), 
 (4, "Ashoka Universal School", 1110, 40, 40, "aus17@gmail.com"), 
 (5, "Calibers English Medium School", 9000, 31, 50, "cems@gmail.com");
 
select * from t_school ;

commit ;

SAVEPOINT savepoint1 ;  

SAVEPOINT savepoint2 ;  

SAVEPOINT savepoint3 ;  

rollback to savepoint3 ;

rollback to savepoint1 ;

rollback ;


-- windows functions

count, max, min, sum, avg, distinct

-- windows analytical function 

row_number(), rank(), dense_rank()


CREATE TABLE employees1 (
    employee_id INT,
    employee_name VARCHAR(100),
    region VARCHAR(50),
    department VARCHAR(50),
    sales DECIMAL
);

-- Insert sample data into the table (100 records)
INSERT INTO employees1 (employee_id, employee_name, region, department, sales)
VALUES
(1, 'Alice', 'North', 'Sales', 5000),
(2, 'Bob', 'North', 'Sales', 7000),
(3, 'Charlie', 'South', 'Marketing', 3000),
(4, 'David', 'North', 'Sales', 4000),
(5, 'Eve', 'South', 'Sales', 6000),
(6, 'Frank', 'South', 'Marketing', 8000),
(7, 'Grace', 'North', 'HR', 2000),
(8, 'Hannah', 'South', 'HR', 3500),
(9, 'Ivy', 'West', 'Sales', 5500),
(10, 'Jack', 'East', 'Sales', 4500),
(11, 'Kylie', 'West', 'HR', 2500),
(12, 'Liam', 'East', 'Sales', 7000),
(13, 'Megan', 'North', 'Marketing', 9000),
(14, 'Nate', 'South', 'HR', 3800),
(15, 'Olivia', 'West', 'Sales', 6000),
(16, 'Paul', 'East', 'Marketing', 5000),
(17, 'Quinn', 'North', 'HR', 2700),
(18, 'Rita', 'South', 'Sales', 8000),
(19, 'Sam', 'West', 'Marketing', 3500),
(20, 'Tom', 'East', 'HR', 2200),
(21, 'Uma', 'North', 'Sales', 7500),
(22, 'Vera', 'South', 'Marketing', 3200),
(23, 'Walter', 'North', 'HR', 1900),
(24, 'Xena', 'East', 'Sales', 5000),
(25, 'Yara', 'South', 'HR', 2700),
(26, 'Zoe', 'West', 'Sales', 6500),
(27, 'Alan', 'East', 'Sales', 4600),
(28, 'Becky', 'South', 'HR', 3100),
(29, 'Carlos', 'North', 'HR', 2900),
(30, 'Diana', 'West', 'Marketing', 7000),
(31, 'Ethan', 'South', 'Sales', 4200),
(32, 'Fiona', 'North', 'Marketing', 8400),
(33, 'Gina', 'East', 'HR', 2800),
(34, 'Howard', 'South', 'Marketing', 7700),
(35, 'Iris', 'West', 'Sales', 6000),
(36, 'Jake', 'East', 'Sales', 5100),
(37, 'Kim', 'South', 'HR', 2600),
(38, 'Leo', 'North', 'Sales', 6400),
(39, 'Maya', 'South', 'Sales', 6800),
(40, 'Nora', 'West', 'Marketing', 5400),
(41, 'Oscar', 'North', 'Sales', 5600),
(42, 'Paula', 'East', 'Marketing', 6200),
(43, 'Quincy', 'South', 'Sales', 7700),
(44, 'Randy', 'West', 'Sales', 4600),
(45, 'Sophie', 'North', 'HR', 2400),
(46, 'Tony', 'South', 'HR', 3300),
(47, 'Ursula', 'West', 'Marketing', 6500),
(48, 'Victor', 'North', 'Sales', 5800),
(49, 'Wendy', 'South', 'Sales', 4000),
(50, 'Xander', 'East', 'Sales', 5300),
(51, 'Yasmine', 'West', 'HR', 2600),
(52, 'Zach', 'South', 'Sales', 5200),
(53, 'Ava', 'North', 'Sales', 7300),
(54, 'Brandon', 'East', 'Sales', 5800),
(55, 'Caitlin', 'West', 'Sales', 4600),
(56, 'Derek', 'North', 'Marketing', 9000),
(57, 'Emily', 'South', 'HR', 3300),
(58, 'Felix', 'East', 'HR', 3500),
(59, 'Graham', 'North', 'HR', 2300),
(60, 'Holly', 'South', 'Sales', 7000),
(61, 'Isaac', 'West', 'HR', 2700),
(62, 'Julia', 'North', 'Sales', 8100),
(63, 'Kevin', 'South', 'Marketing', 5400),
(64, 'Laura', 'East', 'Sales', 6000),
(65, 'Mike', 'North', 'HR', 2200),
(66, 'Nancy', 'West', 'Marketing', 7500),
(67, 'Oscar', 'South', 'HR', 3800),
(68, 'Pamela', 'East', 'Sales', 6700),
(69, 'Quincy', 'North', 'Marketing', 5000),
(70, 'Rachel', 'South', 'Sales', 5700),
(71, 'Steve', 'West', 'Sales', 6900),
(72, 'Tina', 'South', 'Marketing', 6500),
(73, 'Ulysses', 'East', 'HR', 3000),
(74, 'Vince', 'North', 'Sales', 4800),
(75, 'Will', 'South', 'Sales', 4900),
(76, 'Xena', 'West', 'HR', 2900),
(77, 'Yvonne', 'North', 'HR', 3400),
(78, 'Zoe', 'South', 'Sales', 7600),
(79, 'Amy', 'East', 'Sales', 5400),
(80, 'Brad', 'North', 'HR', 2100),
(81, 'Claire', 'South', 'Sales', 6700),
(82, 'Doug', 'West', 'Marketing', 4500),
(83, 'Ella', 'North', 'Sales', 4800),
(84, 'Felicia', 'South', 'HR', 2200),
(85, 'George', 'East', 'Sales', 5100),
(86, 'Hannah', 'West', 'HR', 3800),
(87, 'Ivan', 'South', 'Sales', 5600),
(88, 'Jackie', 'North', 'Sales', 6900),
(89, 'Kimberly', 'South', 'HR', 3100),
(90, 'Landon', 'West', 'Sales', 5300),
(91, 'Monica', 'East', 'Marketing', 6000),
(92, 'Nancy', 'South', 'HR', 2400),
(93, 'Olivia', 'West', 'Sales', 5500),
(94, 'Parker', 'North', 'HR', 3200),
(95, 'Quinn', 'South', 'Sales', 5200),
(96, 'Rachel', 'East', 'Marketing', 5000),
(97, 'Samantha', 'South', 'HR', 3100),
(98, 'Terry', 'North', 'Sales', 4600),
(99, 'Ursula', 'South', 'Sales', 6000),
(100, 'Victor', 'West', 'HR', 4200);

select * from employees1 ;

SELECT  employee_id, employee_name, region, department, sales,
    ROW_NUMBER() OVER (PARTITION BY region, department ORDER BY sales DESC) AS `row_number`
FROM employees1 
ORDER BY region, department, `row_number`;


SELECT 
    employee_id,
    employee_name,
    region,
    department,
    sales,
    rank() OVER (PARTITION BY region, department ORDER BY sales DESC) AS `rank_number`
FROM employees1 where `rank_number` = 1
ORDER BY region, department, `rank_number`;


SELECT 
    employee_id,
    employee_name,
    region,
    department,
    sales,
dense_rank() over(partition by region, department order by sales Desc) as "Rank"
from employees1 ;   


-- Common table expression (cte)

with new_table as (SELECT 
    employee_id,
    employee_name,
    region,
    department,
    sales,
dense_rank() over(partition by region, department order by sales Desc) as "Rank"
from employees1 ) 
select * from new_table 
where `rank` in (1, 2,3 );

SELECT 
    employee_id,
    employee_name,
    region,
    department,
    sales,
    rank() OVER (PARTITION BY region, department ORDER BY sales DESC) AS `rank_number`,
    dense_rank() over(partition by region, department order by sales Desc) as "Rank"
FROM employees1 
ORDER BY region, department, `rank_number`;









