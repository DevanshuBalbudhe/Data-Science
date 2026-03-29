show databases ;
create database class11 ; 
use class11 ;
show tables ;
create table students (`name` varchar(20), age int ) ;
select * from students ;
describe students ;
desc students ;
insert into students value("rahul", 34) ;
insert into students value ("rahul1", 30),
("rahul2", 31),
("rahul3", 32) ;

alter table students
add column course varchar(10) ;

alter table students
add column address varchar(10) after `name` ;


alter table students
add column id int first ;

alter table students
add column DOJ varchar(20)  ;

update students 
set doj = "2025-10-11" ;

alter table students
modify column doj date ;

update students
set id = 2 where name = "rahul1" ;

update students
set id = 3 where age = 31 ;

update students
set address = "ngp" ;

update students
set course = "DS" ;

delete from students
where course = "ds" ;

insert into students value (1, "rahul1","ngp", 30, "ds", "2025-11-10"),
(2, "rahul2","ngp", 31, "ds", "2025-11-11"),
(3, "rahul3","ngp", 32, "ds", "2025-11-12") ;

insert into students (id, name, age, course, doj) 
value (4, "rahul4", 30, "ds", "2025-11-10") ;

insert into students value (5, "rahul5",null, 30, "ds", "2025-11-10") ;

update students
set course = null ;

alter table students
drop column course ;

alter table students
rename column doj to `date` ;

update students
set address = "pune" where name = "Rahul4" ;

update students
set address = "pune" where address is null ;

rename table students to stud ;

truncate table stud ;

select * from stud  ;

drop table stud ;

----------------------------------------------------


-- constraint

not null
default
enum
unique
auto_increment
check
primary key
foreign key
index


create table students (
id int auto_increment primary key, 
`name` varchar(20) not null, 
age int check (age > 18) ,
email varchar(20) unique,
address varchar(20) default "Nagpur",
course enum("DS", "DA", "SAP", "Cloud" )
) ;


insert into students (name, age, email, address, course)
 values ("amit", 19 , "amit3@gmail.com", "pune", "DA") ;
 
 insert into students (name, age, email, course)
 values ("amit", 19 , "amit4@gmail.com", "DA") ;

select * from students ;


















