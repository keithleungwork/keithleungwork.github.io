# CJK special handling

---

## Resource

- A good site to check a character encoding, block library - [https://unicode-table.com/en/23CFE/](https://unicode-table.com/en/23CFE/)

---

## MySQL :

If inserting special character into MySQL.

(such as `𣳾`, it is CJK Unified Ideographs Extension B, according to [here](https://unicode-table.com/en/23CFE/))

Ensure the column charset is correct.

> since v5.5, it is available in the `[utf8mb4](http://dev.mysql.com/doc/en/charset-unicode-utf8mb4.html)`, `[utf16](http://dev.mysql.com/doc/en/charset-unicode-utf16.html)`, `[utf16le](http://dev.mysql.com/doc/en/charset-unicode-utf16le.html)` and `[utf32](http://dev.mysql.com/doc/en/charset-unicode-utf32.html)` character sets.
> 
> 
> It is not available in MySQL's `big5` or `gbk` character sets.
>