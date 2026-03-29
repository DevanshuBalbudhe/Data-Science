use class11 ;

create table if not exists course (
course_id int ,
course_name varchar(50),
course_desc varchar(60),
course_tag varchar(50)) ;


create table if not exists student(
student_id int ,
student_name varchar(30),
student_mobile int ,
student_course_enroll varchar(30),
student_course_id int ) ;

insert into course values(101 , 'fsda' , 'full stack data analytics' , 'Analytics'),
(102 , 'fsds' , 'full stack data analytics' , 'Analytics'),
(103 , 'fsds' , 'full stack data science' , 'DS'),
(104 , 'big data' , 'full stack big data' , 'BD'),
(105 , 'mern' , 'web dev' , 'mern'),
(106 , 'blockchain' , 'full stack blockchain' , 'BC'),
(101 , 'java' , 'full stack java' , 'java'),
(102 , 'testing' , 'full testing ' , 'testing '),
(105 , 'cybersecurity' , 'full stack cybersecurity' , 'cybersecurity'),
(109 , 'c' , 'c language' , 'c'),
(105 , 'c++' , 'C++ language' , 'language') ;

insert into student values(301 , "sudhanshu", 3543453,'yes', 104),
(302 , "sudhanshu", 3543453,'yes', 102),
(301 , "sudhanshu", 3543453,'yes', 105),
(302 , "sudhanshu", 3543453,'yes', 106),
(303 , "sudhanshu", 3543453,'yes', 110),
(304 , "sudhanshu", 3543453,'yes', 103),
(305 , "sudhanshu", 3543453,'yes', 105),
(306 , "sudhanshu", 3543453,'yes', 107),
(306 , "sudhanshu", 3543453,'yes', 108) ;


create table if not exists course_detail (
course_id int ,
course_name varchar(50),
course_desc varchar(60),
course_tag varchar(50)) ;

insert into course_detail values(101 , 'fsda' , 'full stack data analytics' , 'Analytics'),
(102 , 'fsds' , 'full stack data analytics' , 'Analytics'),
(103 , 'fsds' , 'full stack data science' , 'DS'),
(104 , 'big data' , 'full stack big data' , 'BD'),
(105 , 'mern' , 'web dev' , 'mern') ;



select * from course ;

select * from student ;

select course.course_id, student.student_course_id, course.course_name, student.student_name
from course join student 
on course.course_id = student.student_course_id ;

select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c join student s
on c.course_id = s.student_course_id ;


select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c left join student s
on c.course_id = s.student_course_id ;


select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c right join student s
on c.course_id = s.student_course_id ;


select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c cross join student s ;

select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c right join student s
on c.course_id = s.student_course_id 
where c.course_id is null;


select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c inner join student s
on c.course_id = s.student_course_id 
where c.course_id = 105 ;

select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c inner join student s
on c.course_id = s.student_course_id 
where c.course_id in (101, 102, 103, 105, 106) ;

select c.course_id, s.student_course_id, c1.course_id, c.course_name, s.student_name, 
c1.course_tag
from course c join student s
on c.course_id = s.student_course_id 
join course_detail c1 
on c.course_id = c1.course_id  ;






select c.course_id, s.student_course_id, c.course_name, s.student_name
from course c join student s
on c.course_id = s.student_course_id 
where c.course_id = 105 ;



-- Create the employees table
CREATE TABLE employees (
    employee_id INT,
    `name` VARCHAR(50),
    manager_id INT
);

-- Insert demo data into the employees table
INSERT INTO employees (employee_id, `name`, manager_id) VALUES
(1, 'Alice', NULL),
(2, 'Bob', 1),
(3, 'Charlie', 1),
(4, 'David', 2),
(5, 'Eva', 2),
(6, 'Alice1', 4),
(7, 'Bob1', 3),
(8, 'Charlie1', 1),
(9, 'David1', 4),
(10, 'Eva1', null) ;


SELECT 
    e1.`name` AS employee_name,
    e2.`name` AS manager_name
FROM employees e1 inner join employees e2
on e1.manager_id = e2.employee_id;


select course_id, course_name from course
union 
select student_course_id, student_name from student ;

select course_id, course_name from course
union all
select student_course_id, student_name from student ;

select course_id, course_name from course
union all
select student_name, student_course_id  from student ;

select course_id, course_name from course
union 
select student_name, student_course_id  from student 
union
select employee_id, `name` from employees ;
