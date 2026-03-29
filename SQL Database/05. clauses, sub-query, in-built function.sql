use class11 ;

CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10, 2),
    stock INT,
    supplier VARCHAR(100),
    release_date DATE
);

INSERT INTO products (product_name, category, price, stock, supplier, release_date) VALUES
('Laptop', 'Electronics', 999.99, 25, 'TechCorp', '2023-01-15'),
('Smartphone', 'Electronics', 799.99, 50, 'PhoneInc', '2023-02-10'),
('Tablet', 'Electronics', 499.99, 30, 'GadgetWorld', '2023-03-05'),
('Headphones', 'Accessories', 199.99, 100, 'SoundMaster', '2023-01-20'),
('Charger', 'Accessories', 29.99, 200, 'PowerUp', '2023-02-15'),
('Monitor', 'Electronics', 299.99, 15, 'DisplayPro', '2023-04-01'),
('Keyboard', 'Accessories', 89.99, 75, 'KeyTech', '2023-01-25'),
('Mouse', 'Accessories', 49.99, 80, 'ClickIt', '2023-02-28'),
('Smartwatch', 'Wearables', 249.99, 40, 'WearableTech', '2023-03-20'),
('Camera', 'Electronics', 599.99, 20, 'PhotoGear', '2023-01-30'),
('Printer', 'Electronics', 159.99, 35, 'PrintMasters', '2023-03-10'),
('External Hard Drive', 'Storage', 119.99, 60, 'DataSafe', '2023-02-01'),
('USB Flash Drive', 'Storage', 19.99, 150, 'QuickStore', '2023-03-15'),
('Router', 'Networking', 89.99, 45, 'NetGear', '2023-01-10'),
('Webcam', 'Accessories', 69.99, 55, 'VisionTech', '2023-02-20'),
('Bluetooth Speaker', 'Audio', 149.99, 70, 'AudioWave', '2023-03-30'),
('Gaming Console', 'Electronics', 499.99, 10, 'GameBox', '2023-04-10'),
('Game Controller', 'Accessories', 59.99, 80, 'GameGear', '2023-01-05'),
('VR Headset', 'Wearables', 399.99, 5, 'VirtualVisions', '2023-04-15'),
('Smart TV', 'Electronics', 899.99, 8, 'VisionTech', '2023-03-25'),
('Streaming Device', 'Electronics', 49.99, 60, 'Streamline', '2023-02-18'),
('Dishwasher', 'Home Appliances', 899.99, 12, 'HomeTech', '2023-01-12'),
('Washing Machine', 'Home Appliances', 499.99, 10, 'CleanHome', '2023-01-29'),
('Refrigerator', 'Home Appliances', 999.99, 6, 'CoolTech', '2023-03-02'),
('Air Conditioner', 'Home Appliances', 599.99, 7, 'ClimateControl', '2023-02-05'),
('Microwave', 'Home Appliances', 199.99, 25, 'QuickCook', '2023-03-12'),
('Electric Kettle', 'Kitchen', 39.99, 80, 'KitchenAid', '2023-01-22'),
('Blender', 'Kitchen', 79.99, 50, 'BlendMaster', '2023-02-02'),
('Coffee Maker', 'Kitchen', 129.99, 45, 'BrewTime', '2023-01-18'),
('Toaster', 'Kitchen', 49.99, 100, 'ToastPerfect', '2023-03-04'),
('Vacuum Cleaner', 'Home Appliances', 249.99, 20, 'CleanSweep', '2023-01-08'),
('Rice Cooker', 'Kitchen', 59.99, 30, 'CookEase', '2023-02-22'),
('Dish Rack', 'Kitchen', 19.99, 150, 'SpaceSaver', '2023-04-03'),
('Table Lamp', 'Furniture', 39.99, 90, 'LightHouse', '2023-01-15'),
('Office Chair', 'Furniture', 149.99, 40, 'ComfortSeating', '2023-02-10'),
('Desk', 'Furniture', 199.99, 25, 'WorkSpace', '2023-03-05'),
('Bookshelf', 'Furniture', 99.99, 15, 'BookHaven', '2023-01-20'),
('Dining Table', 'Furniture', 299.99, 8, 'HomeStyle', '2023-02-15'),
('Sofa', 'Furniture', 799.99, 5, 'ComfortCouch', '2023-03-01'),
('Bed', 'Furniture', 599.99, 7, 'RestEasy', '2023-02-28'),
('Nightstand', 'Furniture', 79.99, 50, 'SleepWell', '2023-01-12'),
('Duvet Cover', 'Bedding', 29.99, 100, 'SoftSleep', '2023-03-15'),
('Pillow', 'Bedding', 19.99, 120, 'CloudRest', '2023-01-05'),
('Curtains', 'Home Decor', 39.99, 60, 'WindowTreats', '2023-02-22'),
('Wall Art', 'Home Decor', 89.99, 30, 'ArtisticVibes', '2023-04-05'),
('Vase', 'Home Decor', 24.99, 150, 'FloralEssence', '2023-01-15'),
('Rug', 'Home Decor', 99.99, 20, 'CozyHome', '2023-03-20'),
('Cookware Set', 'Kitchen', 149.99, 15, 'ChefPro', '2023-01-28'),
('Cutlery Set', 'Kitchen', 79.99, 35, 'SharpEdge', '2023-02-12'),
('Dinner Plates', 'Kitchen', 49.99, 60, 'PlatePerfect', '2023-03-09'),
('Glassware Set', 'Kitchen', 29.99, 80, 'ClearView', '2023-04-01'),
('Garden Hose', 'Garden', 19.99, 90, 'GardenTools', '2023-02-18'),
('Lawn Mower', 'Garden', 299.99, 12, 'MowMaster', '2023-03-04'),
('Shovel', 'Garden', 24.99, 70, 'DigIt', '2023-01-10'),
('Rake', 'Garden', 14.99, 100, 'GardenTools', '2023-04-20'),
('Fertilizer', 'Garden', 9.99, 200, 'GrowFast', '2023-02-28');

select * from products ;

-- where clause

SELECT * FROM products WHERE price > 200;

SELECT * FROM products WHERE category = 'Electronics';

SELECT count(*) FROM products WHERE release_date > '2023-03-01';

SELECT * FROM products WHERE price > 500  and stock > 5 ;

SELECT price, stock FROM products WHERE price between 50 and 200 ;

SELECT *  FROM products WHERE stock in (25, 20, 12, 5);

-- group by 

SELECT category, COUNT(*) AS product_count FROM products GROUP BY category ;

SELECT supplier, AVG(price) AS average_price FROM products GROUP BY supplier ;

SELECT category, SUM(stock) AS total_stock FROM products GROUP BY category;

SELECT month(release_date) AS release_year, category, COUNT(*) AS product_count  FROM products 
GROUP BY release_year, category;

-- having 

SELECT category, COUNT(*) AS product_count FROM products GROUP BY category HAVING COUNT(*) > 5;

SELECT supplier, AVG(price) AS average_price FROM products GROUP BY supplier 
HAVING AVG(price) > 100;

SELECT supplier, MAX(price) AS max_price FROM products GROUP BY supplier 
HAVING MAX(price) < 200;


-- order by

SELECT * FROM products ORDER BY price ASC;

SELECT * FROM products ORDER BY stock DESC;

SELECT * FROM products ORDER BY category ASC, price DESC;


-- limit

SELECT * FROM products LIMIT 10;

SELECT * FROM products ORDER BY id DESC LIMIT 10;

SELECT * FROM products LIMIT 10 offset 10;


-- distinct

SELECT DISTINCT category FROM products;

SELECT category, AVG(price) AS average_price, COUNT(*) AS product_count FROM products
WHERE stock > 10  -- Filter for products that are in stock
GROUP BY category  -- Group results by category
HAVING COUNT(*) > 5  -- Only include categories with more than 5 products
ORDER BY average_price DESC   -- Order the results by average price, highest first
limit 2 offset 1;

SELECT category as cat, AVG(price) AS average_price, COUNT(*) AS product_count FROM products
WHERE stock > 10  -- Filter for products that are in stock
GROUP BY cat  -- Group results by category
HAVING COUNT(*) > 5  -- Only include categories with more than 5 products
ORDER BY average_price DESC   -- Order the results by average price, highest first
limit 2 offset 1;

-- sub query

-- Find Categories with Average Price Above a Certain Value
SELECT category, sum(price) FROM products GROUP BY category HAVING AVG(price) > 
( SELECT AVG(price) FROM products );

-- Count Products in Categories with Low Stock
SELECT category, COUNT(*) AS product_count FROM products WHERE category IN 
( SELECT category FROM products GROUP BY category HAVING SUM(stock) > 100 )
GROUP BY category;


-- List Suppliers with More Products than the Average
SELECT supplier FROM products GROUP BY supplier HAVING COUNT(*) > 
(SELECT AVG(product_count) FROM (SELECT COUNT(*) AS product_count FROM products GROUP BY supplier)
 AS avg_count );
 
 
-- In-built function 

### 1. **`CONCAT()`**
-- Concatenates two or more strings together.

SELECT CONCAT('Hello ', 'World');

select concat(supplier,"--", category ) from products ;

### 2. **`CONCAT_WS()`**
-- Concatenates two or more strings with a separator between them.

SELECT CONCAT_WS('-', '2025', '01', '14');

### 3. **`CHAR_LENGTH()`**
-- Returns the number of characters in a string (character count, not byte count).

SELECT CHAR_LENGTH('Hello');

### 4. **`LENGTH()`**
-- Returns the length of a string in bytes (character length in a multibyte character set may differ from the byte length).

SELECT LENGTH('Hello');


### 5. **`LOWER()`**
-- Converts all characters in a string to lowercase.

SELECT LOWER('HELLO');

### 6. **`UPPER()`**
-- Converts all characters in a string to uppercase.

SELECT UPPER('hello');

### 7. **`TRIM()`**
-- Removes leading and trailing spaces from a string.

SELECT TRIM('  Hello  ');

### 8. **`LTRIM()`**
-- Removes leading spaces from a string.

SELECT LTRIM('  Hello');

### 9. **`RTRIM()`**
-- Removes trailing spaces from a string.

SELECT RTRIM('Hello  ');
 
### 10. **`SUBSTRING()`**
-- Extracts a substring from a string starting at a specified position.

SELECT SUBSTRING('Hello World', 3, 5); 

### 11. **`LEFT()`**
-- Returns the leftmost `n` characters from a string.

SELECT LEFT('Hello World', 5);
-- Output: 'Hello'


### 12. **`RIGHT()`**
-- Returns the rightmost `n` characters from a string.

SELECT RIGHT('Hello World', 5);
-- Output: 'World'

### 13. **`REPLACE()`**
-- Replaces all occurrences of a substring within a string with another substring.

SELECT REPLACE('Hello World', 'World', 'MySQL');


### 14. **`INSTR()`**
-- Returns the position of the first occurrence of a substring in a string (returns 0 if not found).

SELECT INSTR('Hello World', 'World'); 

### 16. **`REVERSE()`**
-- Reverses the characters in a string.

SELECT REVERSE('Hello');

### 18. **`FIELD()`**
-- Returns the index of a value in a list of arguments. If the value is not found, it returns 0.

SELECT FIELD('B', 'A', 'B','C');

### 19. **`LPAD()`**
-- Pads the left side of a string with a specified character to a given length.

SELECT LPAD('Hello', 10, '-');

### 20. **`RPAD()`**
-- Pads the right side of a string with a specified character to a given length.
 
SELECT RPAD('Hello', 10, '-');


### 1. **`ABS()`**
Returns the absolute value of a number.
 
SELECT ABS(-10);  
-- Output: 10
 

### 2. **`ACOS()`**
Returns the arc cosine of a number (in radians).
 
SELECT ACOS(0.5);  
-- Output: 1.0471975511965979 (radian value)
 

### 3. **`ASIN()`**
Returns the arc sine of a number (in radians).
 
SELECT ASIN(0.5);  
-- Output: 0.5235987755982989 (radian value)
 

### 4. **`ATAN()`**
Returns the arc tangent of a number (in radians).
 
SELECT ATAN(1);  
-- Output: 0.7853981633974483 (radian value)
 

### 5. **`ATAN2()`**
Returns the arc tangent of the two numbers (x, y), where `y/x` is the tangent of the angle.
 
SELECT ATAN2(1, 1);  
-- Output: 0.7853981633974483 (radian value)
 

### 6. **`CEIL()` or `CEILING()`**
Returns the smallest integer greater than or equal to a number.
```sql
SELECT CEIL(10.5);  
-- Output: 11
```

### 7. **`COS()`**
Returns the cosine of a number (in radians).
 
SELECT COS(PI()/3);  
-- Output: 0.5
 

### 8. **`COT()`**
Returns the cotangent of a number (in radians).
 
SELECT COT(PI()/4);  
-- Output: 1.0
 

### 9. **`DIV()`**
Divides two integers and returns the integer part of the quotient.
 
SELECT 10 DIV 3;  
-- Output: 3
 

### 10. **`EXP()`**
Returns `e` raised to the power of a number (Euler’s number).
 
SELECT EXP(1);  
-- Output: 2.718281828459045 (e^1)
 

### 11. **`FLOOR()`**
Returns the largest integer less than or equal to a number.
 
SELECT FLOOR(10.5);  
-- Output: 10
 

### 12. **`GREATEST()`**
Returns the largest value in a list of expressions.
 
SELECT GREATEST(3, 5, 7, 2);  
-- Output: 7
 

### 13. **`LEAST()`**
Returns the smallest value in a list of expressions.
 
SELECT LEAST(3, 5, 7, 2);  
-- Output: 2
 

### 14. **`LN()`**
Returns the natural logarithm (base `e`) of a number.
 
SELECT LN(10);  
-- Output: 2.302585092994046 (log_e(10))
 

### 15. **`LOG()`**
Returns the natural logarithm of a number or logarithm to a specified base.
 
SELECT LOG(10);  
-- Output: 2.302585092994046 (log_e(10))

-- With base specified
SELECT LOG(10, 2);  
-- Output: 3.321928094887362 (log_2(10))
 

### 16. **`MOD()`**
Returns the remainder of a division.
 
SELECT MOD(10, 3);  
-- Output: 1
 

### 17. **`PI()`**
Returns the value of `π` (Pi).
 
SELECT PI();  
-- Output: 3.141592653589793
 

### 18. **`POW()` or `POWER()`**
Returns the value of the first number raised to the power of the second number.
 
SELECT POW(2, 3);  
-- Output: 8

SELECT POWER(2, 3);  
-- Output: 8
 

### 19. **`RAND()`**
Returns a random floating-point value between 0 and 1.
 
SELECT RAND();  
-- Output: (random value between 0 and 1)
 

### 20. **`ROUND()`**
Rounds a number to the specified number of decimal places.
 
SELECT ROUND(12.34567, 2);  
-- Output: 12.35
 

### 21. **`SIGN()`**
Returns the sign of a number: 1 for positive, 0 for zero, and -1 for negative.
 
SELECT SIGN(5);  
-- Output: 1

SELECT SIGN(-5);  
-- Output: -1

SELECT SIGN(0);  
-- Output: 0
 

### 22. **`SIN()`**
Returns the sine of a number (in radians).
 
SELECT SIN(PI()/6);  
-- Output: 0.5
 

### 23. **`SQRT()`**
Returns the square root of a number.
 
SELECT SQRT(16);  
-- Output: 4
 

### 24. **`TAN()`**
Returns the tangent of a number (in radians).
 
SELECT TAN(PI()/4);  
-- Output: 1
 

### 25. **`TRUNCATE()`**
Truncates a number to a specified number of decimal places without rounding.
 
SELECT TRUNCATE(12.34567, 2);  
-- Output: 12.34
 

 

 

 
 

### 29. **`DEGREES()`**
Converts a radian value to degrees.
 
SELECT DEGREES(PI()/2);  
-- Output: 90
 

### 30. **`RADIANS()`**
Converts a degree value to radians.
 
SELECT RADIANS(180);  
-- Output: 3.141592653589793


### 1. **`CURDATE()`**
Returns the current date in `YYYY-MM-DD` format.
 
SELECT CURDATE();  
-- Output: '2025-01-14'
 

### 2. **`CURTIME()`**
Returns the current time in `HH:MM:SS` format.
 
SELECT CURTIME();  
-- Output: '12:45:30' (example time)
 

### 3. **`NOW()`**
Returns the current date and time in `YYYY-MM-DD HH:MM:SS` format.
 
SELECT NOW();  
-- Output: '2025-01-14 12:45:30'
 

### 4. **`UTC_DATE()`**
Returns the current UTC date in `YYYY-MM-DD` format.
 
SELECT UTC_DATE();  
-- Output: '2025-01-14'
 

### 5. **`UTC_TIME()`**
Returns the current UTC time in `HH:MM:SS` format.
 
SELECT UTC_TIME();  
-- Output: '12:45:30' (example UTC time)
 

### 6. **`UTC_TIMESTAMP()`**
Returns the current UTC date and time in `YYYY-MM-DD HH:MM:SS` format.
 
SELECT UTC_TIMESTAMP();  
-- Output: '2025-01-14 12:45:30'
 

### 7. **`DATE()`**
Extracts the date part from a `DATETIME` or `TIMESTAMP` value.
 
SELECT DATE('2025-01-14 12:45:30');  
-- Output: '2025-01-14'
 

### 8. **`TIME()`**
Extracts the time part from a `DATETIME` or `TIMESTAMP` value.
 
SELECT TIME('2025-01-14 12:45:30');  
-- Output: '12:45:30'
 

### 9. **`YEAR()`**
Extracts the year from a date.
 
SELECT YEAR('2025-01-14');  
-- Output: 2025
 

### 10. **`MONTH()`**
Extracts the month from a date.
 
SELECT MONTH('2025-01-14');  
-- Output: 1 (January)
 

### 11. **`DAY()`**
Extracts the day of the month from a date.
 
SELECT DAY('2025-01-14');  
-- Output: 14
 

### 12. **`DAYOFMONTH()`**
Returns the day of the month (1-31) for a given date.
 
SELECT DAYOFMONTH('2025-01-14');  
-- Output: 14
 

### 13. **`DAYOFWEEK()`**
Returns the day of the week (1-7, where 1 is Sunday and 7 is Saturday).
 
SELECT DAYOFWEEK('2025-01-14');  
-- Output: 3 (Tuesday)
 

### 14. **`DAYOFYEAR()`**
Returns the day of the year (1-366).
 
SELECT DAYOFYEAR('2025-11-14');  
-- Output: 14
 

### 15. **`WEEK()`**
Returns the week of the year (0-52).
 
SELECT WEEK('2025-01-14');  
-- Output: 2
 

### 16. **`WEEKDAY()`**
Returns the weekday index (0-6, where 0 is Monday and 6 is Sunday).
 
SELECT WEEKDAY('2025-01-14');  
-- Output: 1 (Tuesday)
 

### 17. **`HOUR()`**
Extracts the hour part from a time or `DATETIME` value.
 
SELECT HOUR('2025-01-14 12:45:30');  
-- Output: 12
 

### 18. **`MINUTE()`**
Extracts the minute part from a time or `DATETIME` value.
 
SELECT MINUTE('2025-01-14 12:45:30');  
-- Output: 45
 

### 19. **`SECOND()`**
Extracts the second part from a time or `DATETIME` value.
 
SELECT SECOND('2025-01-14 12:45:30');  
-- Output: 30
 

### 20. **`DATE_ADD()`**
Adds a time interval to a date.
 
SELECT DATE_ADD('2025-01-14', INTERVAL -10 DAY);  
-- Output: '2025-01-15'
 

### 21. **`DATE_SUB()`**
Subtracts a time interval from a date.
 
SELECT DATE_SUB('2025-01-14', INTERVAL -1 DAY);  
-- Output: '2025-01-13'
 

### 22. **`DATEDIFF()`**
Returns the difference in days between two dates.
 
SELECT DATEDIFF('2025-01-14', '2025-01-01');  
-- Output: 13
 

### 23. **`TIMESTAMPDIFF()`**
Returns the difference between two date or time expressions in a specified unit (e.g., days, months, years).
 
SELECT TIMESTAMPDIFF(DAY, '2025-01-01', '2025-01-14');  
-- Output: 13
 

### 24. **`TIMESTAMPADD()`**
Adds an interval to a date or time value.
 
SELECT TIMESTAMPADD(DAY, 10, '2025-01-14');  
-- Output: '2025-01-24'
 

### 25. **`STR_TO_DATE()`**
Converts a string to a date based on a specified format.
 
SELECT STR_TO_DATE('2025-01-14', '%Y-%b-%d');  
-- Output: '2025-01-14'
 

### 26. **`DATE_FORMAT()`**
Formats a date or time according to a specified format.
 
SELECT DATE_FORMAT('2025-01-14', '%W, %M %d, %Y');  
-- Output: 'Tuesday, January 14, 2025'
 

 

 

### 29. **`ISDATE()`**
Checks if the expression is a valid date.
 
SELECT ISDATE('2025-01-14');  
-- Output: 1 (valid date)

SELECT ISDATE('2025-02-30');  
-- Output: 0 (invalid date)
 

### 30. **`TIMEDIFF()`**
Returns the difference between two time or `DATETIME` values.
 
SELECT TIMEDIFF('2025-01-14 14:00:00', '2025-01-14 12:45:30');  
-- Output: '01:14:30'
 

### 31. **`TIME_FORMAT()`**
Formats a time value according to a specified format.
 
SELECT TIME_FORMAT('12:45:30', '%h:%i:%s %p');  
-- Output: '12:45:30 PM'
 


 

