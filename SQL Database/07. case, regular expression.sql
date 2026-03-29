select * from products ;

select product_name, stock,
case
	when stock > 150 then "high"
    when stock < 80 then "low"
    else "medium"
end as stock_flag
from products ;   


select product_name, stock, category,
case
	when category > "M" then "M"
    when category < "H" then "H"
    else "Others"
end as stock_flag
from products ;   


select product_name, stock, category,
case
	when category = "Electronics" then "ele"
    when category = "Accessories" then "Accs"
	when category = "Networking" then "N/W"
    else "Others"
end as cat_flag
from products ;   


select product_name, stock, category,
case
	when category = "Electronics" then "ele"
    when category = "Accessories" then "Accs"
	when category = "Networking" then "N/W"
    else category
end as cat_flag
from products ;   

alter table products
add column stock_cat varchar(10) ;


update products
set stock_cat = case
					when stock > 150 then "high"
					when stock < 80 then "low"
					else "medium"
				end ;

delete from products
where product_name in ("keyborad", "mouse", "camera") ;


delete from products
where product_name = case
						when product_name = "keyboard" then "keyboard"
                        when product_name = "printer" then "printer"
                        end ;


select * from products ;


-- regular expression

select * from laptop_data ;

select * from laptop_data where `cpu` regexp "i3" ;

select * from laptop_data where `gpu` regexp "^G" ;

select * from laptop_data where `gpu` regexp "^[gi]" ;

select * from laptop_data where `cpu` regexp "x$" ;

select * from laptop_data where `cpu` regexp "[hx]$" ;

select * from laptop_data where price_inr regexp "^4400" ;

select * from laptop_data where brand rlike "^m|t$" ;
select * from laptop_data where brand rlike "^m|^a" ;

select * from laptop_data where brand rlike ".ce.." ;

select * from laptop_data where `cpu` like "%h" ;

select * from laptop_data where `cpu` like "I5%" ;

select * from laptop_data where `gpu` like "G%165%B" ;

select * from laptop_data where `sku` like "%004__" ;





  
    