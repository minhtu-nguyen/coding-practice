SHOW TABLES;
DESCRIBE table;
SHOW CREATE TABLE servers; 
SHOW COLUMNS FROM servers;

-- The lesson queries are reproduced below for convenient copy/paste into the terminal. 

-- Query 1
CREATE DATABASE MovieIndustry;

-- Query 2
CREATE DATABASE IF NOT EXISTS MovieIndustry;

-- Query 3
SHOW DATABASES;

-- Query 4
DROP DATABASE MovieIndustry;

-- ***Create table
CREATE TABLE tableName (

col1 <dataType> <Restrictions>,

col2 <dataType> <Restrictions>,

col3 <dataType> <Restrictions>,

<Primary Key or Index definitions>);

-- Inspect a column
DESC Actors;

-- Create table
CREATE TABLE IF NOT EXISTS Actors (
Id INT AUTO_INCREMENT,
FirstName VARCHAR(20) NOT NULL,
SecondName VARCHAR(20) NOT NULL,
DoB DATE NOT NULL,
Gender ENUM('Male','Female','Other') NOT NULL,
MaritalStatus ENUM('Married', 'Divorced', 'Single', 'Unknown') DEFAULT "Unknown",
NetWorthInMillions DECIMAL NOT NULL,
PRIMARY KEY (Id));

-- Temporary Table
CREATE TEMPORARY TABLE tableName (

col1 <dataType> <Restrictions>,

col2 <dataType> <Restrictions>,

col3 <dataType> <Restrictions>,

<Primary Key or Index definitions>);

-- Collation & Character Sets
SHOW CHARACTER SET;
SHOW COLLATION;
SHOW VARIABLES LIKE "c%";

-- Insert
INSERT INTO table (col1, col2 … coln)

VALUES (val1, val2, … valn);

-- Query 1
INSERT INTO Actors ( 
FirstName, SecondName, DoB, Gender, MaritalStatus, NetworthInMillions) 
VALUES ("Brad", "Pitt", "1963-12-18", "Male", "Single", 240.00);

-- Query 2
INSERT INTO Actors ( 
FirstName, SecondName, DoB, Gender, MaritalStatus, NetworthInMillions) 
VALUES 
("Jennifer", "Aniston", "1969-11-02", "Female", "Single", 240.00),
("Angelina", "Jolie", "1975-06-04", "Female", "Single", 100.00),
("Johnny", "Depp", "1963-06-09", "Male", "Single", 200.00);

-- Query 3
INSERT INTO Actors 
VALUES (DEFAULT, "Dream", "Actress", "9999-01-01", "Female", "Single", 000.00);

-- Query 4
INSERT INTO Actors VALUES (NULL, "Reclusive", "Actor", "1980-01-01", "Male", "Single", DEFAULT);

-- Query 5
INSERT INTO Actors () VALUES ();

-- ##Querying Data
-- Query 1
SELECT * from Actors;

-- Query 2
SELECT <columns> FROM <TableName>

-- LIKE operator
-- Query 1
SELECT * from Actors WHERE FirstName LIKE "Jen%";

-- Query 2
SELECT * from Actors where FirstName LIKE "Jennifer%";

-- Query 3
SELECT * from Actors where FirstName LIKE "%";

-- Query 4
SELECT * from Actors WHERE FirstName LIKE "_enn%";

-- ## Combining Conditionals
SELECT * FROM Actors WHERE (FirstName > 'B' AND FirstName < 'J') OR (SecondName >'I' AND SecondName < 'K');

-- Query 4
SELECT * FROM Actors WHERE NOT(FirstName > "B" OR NetWorthInMillions > 200);

-- Query 5
SELECT * FROM Actors WHERE NOT NetWorthInMillions = 200;

-- Query 6
SELECT * FROM Actors WHERE (NOT NetWorthInMillions) = 200;

-- Query 7
SELECT * FROM Actors WHERE FirstName > "B" XOR NetWorthInMillions > 200;


-- #ORDER
SELECT * FROM Actors ORDER BY NetWorthInMillions DESC, FirstName ASC;

SELECT * FROM Actors ORDER BY BINARY FirstName;

SELECT * FROM Actors ORDER BY CAST(NetWorthInMillions AS CHAR);

-- #LIMIT
SELECT FirstName, SecondName from Actors ORDER BY NetWorthInMillions DESC LIMIT 3;

SELECT FirstName, SecondName from Actors ORDER BY NetWorthInMillions DESC LIMIT 4 OFFSET 3;

-- #DELETE
DELETE FROM Actors WHERE Gender="Male";

DELETE FROM Actors ORDER BY NetWorthInMillions DESC LIMIT 3;

-- #TRUNCATE
TRUNCATE table;

-- #UPDATE
UPDATE Actors SET NetWorthInMillions=5 ORDER BY FirstName LIMIT 3;

-- #INDEX
-- Query 1
SHOW INDEX FROM Actors;

-- Query 2 Exact
ANALYZE TABLE Actors;
SHOW INDEX FROM Actors;


-- ALTERATION
ALTER TABLE Actors CHANGE FirstName First_Name varchar(120);
ALTER TABLE Actors MODIFY First_Name varchar(20) DEFAULT "Anonymous";
ALTER TABLE Actors ADD MiddleName 
varchar(100);
ALTER TABLE Actors DROP MiddleName;
ALTER TABLE Actors ADD MiddleName varchar(100) FIRST;
ALTER TABLE Actors DROP MiddleName, ADD Middle_Name varchar(100);

ALTER TABLE Actors ADD INDEX nameIndex (FirstName);
ALTER TABLE Actors DROP INDEX nameIndex;

ALTER TABLE Actors RENAME ActorsTable;
DROP TABLE IF EXISTS ActorsTable;
DROP TABLE IF EXISTS ActorsTable;DROP DATABASE IF EXISTS MovieIndustry;

-- ALIAS
SELECT FirstName AS PopularName from Actors;
SELECT CONCAT(FirstName,' ', SecondName) AS FullName FROM Actors ORDER BY FullName;
SELECT FirstName FROM Actors AS tbl WHERE tbl.FirstName='Brad' AND tbl.NetWorthInMillions > 200;
SELECT tbl.FirstName FROM Actors AS tbl WHERE tbl.FirstName='Brad' AND tbl.NetWorthInMillions > 200;

SELECT t1.FirstName, t1.NetworthInMillions
FROM Actors AS t1,
Actors AS t2
WHERE t1.NetworthInMillions = t2.NetworthInMillions
AND t1.Id != t2.Id;

-- DISTINCT
SELECT DISTINCT MaritalStatus, FirstName from Actors;

-- AGGREGATE
-- Query 1
SELECT COUNT(*) FROM Actors;

-- Query 2
SELECT SUM(NetworthInMillions) FROM Actors;

-- Query 3
SELECT AVG(NetWorthInMillions) FROM Actors;

-- Query 4
SELECT MIN(NetWorthInMillions) FROM Actors;

-- Query 5
SELECT MAX(NetWorthInMillions) FROM Actors;

-- Query 6
SELECT STDDEV(NetWorthInMillions) FROM Actors;

-- GROUP
SELECT Gender FROM Actors GROUP BY Gender;

SELECT MaritalStatus, AVG(NetworthInMillions) FROM Actors GROUP BY MaritalStatus ORDER BY MaritalStatus ASC;

-- HAVING
SELECT MaritalStatus, AVG(NetworthInMillions) AS NetWorth 
FROM Actors 
GROUP BY MaritalStatus 
HAVING MaritalStatus='Married';

SELECT MaritalStatus, AVG(NetworthInMillions) AS NetWorth 
FROM Actors WHERE MaritalStatus='Married' 
GROUP BY MaritalStatus;

-- INNER JOIN
SELECT FirstName, SecondName, AssetType, URL
FROM Actors 
INNER JOIN DigitalAssets  
ON Actors.Id = DigitalAssets.ActorID;

SELECT FirstName, SecondName, AssetType, URL 
FROM Actors 
INNER JOIN DigitalAssets 
USING(Id); -- Same name for ID column

SELECT FirstName, SecondName, AssetType, URL 
FROM Actors, DigitalAssets 
WHERE ActorId=Id; -- Possible not using INNER JOIN

-- LEFT JOIN
SELECT FirstName, SecondName, AssetType, URL
FROM DigitalAssets 
LEFT JOIN Actors
ON Actors.Id = DigitalAssets.ActorID;

-- RIGHT JOIN
SELECT FirstName, SecondName, AssetType, URL
FROM Actors 
RIGHT JOIN DigitalAssets  
ON Actors.Id = DigitalAssets.ActorID;

-- UNION
SELECT FirstName, Id FROM Actors 
UNION 
SELECT FirstName FROM Actors; -- Bad

SELECT FirstName, Id FROM Actors 
UNION 
SELECT FirstName, null FROM Actors; -- OK

(SELECT CONCAT(FirstName, ' ', SecondName) AS "Actor Name"  
FROM Actors  
ORDER BY NetworthInMillions DESC  LIMIT 2)  
UNION  
(SELECT NetworthInMillions 
FROM Actors 
ORDER BY NetworthInMillions ASC); -- Not ordered, missing LIMIT

(SELECT CONCAT(FirstName, ' ', SecondName) AS "Actor Name"  
FROM Actors  
ORDER BY NetworthInMillions DESC  LIMIT 2)  
UNION  
(SELECT NetworthInMillions 
FROM Actors 
ORDER BY NetworthInMillions ASC LIMIT 3); -- OK

-- NESTED QUERIES
SELECT FirstName 
FROM Actors 
INNER JOIN DigitalAssets 
ON ActorId = Id 
WHERE LastUpdatedOn = (SELECT MAX(LastUpdatedOn) 
    FROM DigitalAssets);

SELECT FirstName, SecondName
FROM Actors
WHERE Id = ANY (SELECT ActorId
             FROM DigitalAssets
             WHERE AssetType = 'Facebook');

SELECT FirstName, SecondName
FROM Actors
WHERE Id IN (SELECT ActorId
             FROM DigitalAssets
             WHERE AssetType = 'Facebook');

SELECT FirstName, SecondName 
FROM Actors 
WHERE NetworthInMillions > ALL (SELECT NetworthInMillions 
                             FROM Actors
                             WHERE FirstName LIKE "j%");

SELECT FirstName
FROM Actors 
WHERE (Id, MONTH(DoB), DAY(DoB))
IN ( SELECT ActorId, MONTH(LastUpdatedOn), DAY(LastUpdatedOn)
     FROM DigitalAssets);

SELECT FirstName, AssetType, LastUpdatedOn 
FROM Actors 
INNER JOIN (SELECT ActorId, AssetType, LastUpdatedOn 
            FROM DigitalAssets) AS tbl 
ON ActorId = Id
WHERE FirstName = "Kim"
ORDER BY LastUpdatedOn DESC LIMIT 1;

-- EXISTS Operator
SELECT *
FROM Actors
WHERE EXISTS ( SELECT * 
            FROM DigitalAssets
            WHERE BINARY URL LIKE "%clooney%"); 

-- Correlated Queries
SELECT FirstName
FROM Actors 
INNER JOIN DigitalAssets
ON Id = ActorId
WHERE URL LIKE CONCAT("%",FirstName,"%") 
AND AssetType="Twitter";

SELECT FirstName
FROM Actors
WHERE EXISTS (SELECT URL 
              FROM DigitalAssets
              WHERE URL LIKE CONCAT("%",FirstName,"%") 
              AND AssetType="Twitter");

--  Multi-Table Delete
DELETE Actors, DigitalAssets   -- Mention tables to delete rows from
FROM Actors   -- The inner join creates a derived table 
           -- with matching rows from both tables    
INNER JOIN DigitalAssets
ON Actors.Id = DigitalAssets.ActorId
WHERE AssetType = "Twitter";


DELETE FROM Actors, DigitalAssets
USING Actors        -- newer syntax
INNER JOIN DigitalAssets
ON Actors.Id = DigitalAssets.ActorId
WHERE AssetType = "Twitter";

--  Multi-Table Update
UPDATE 
Actors INNER JOIN DigitalAssets 
ON Id = ActorId 
SET FirstName = UPPER(FirstName), SecondName = UPPER(SecondName), URL = UPPER(URL) 
WHERE AssetType = "Facebook";

-- INSERT
INSERT INTO Names(name) 
SELECT SecondName FROM Actors;

INSERT IGNORE INTO Names(name) 
SELECT SecondName 
FROM Actors WHERE Id = 1;

CREATE TABLE MyTempTable SELECT * FROM Actors;
CREATE TABLE CopyOfActors LIKE Actors;

-- REPLACE
REPLACE INTO
Actors (Id, FirstName, SecondName,DoB, Gender, MaritalStatus, NetworthInMillions)
VALUES (3, "George", "Clooney", "1961-05-06","Male", "Married", 500.00);

-- View
CREATE VIEW DigitalAssetCount AS 
SELECT ActorId, COUNT(AssetType) AS NumberOfAssets 
FROM DigitalAssets
GROUP BY ActorId;

SELECT * FROM DigitalAssetCount;

CREATE OR REPLACE VIEW ActorsTwitterAccounts AS
SELECT CONCAT(FirstName, ' ', SecondName) AS ActorName, URL
FROM Actors
INNER JOIN DigitalAssets  
ON Actors.Id = DigitalAssets.ActorID 
WHERE AssetType = 'Twitter';

CREATE VIEW ActorDetails (ActorName, Age, MaritalStatus, NetWorthInMillions) AS
SELECT CONCAT(FirstName,' ',SecondName) AS ActorName, 
    TIMESTAMPDIFF(YEAR, DoB, CURDATE()) AS Age, 
    MaritalStatus, NetWorthInMillions 
FROM Actors;

UPDATE ActorView 
SET 
NetWorthInMillions = 250 
WHERE 
Id =1;

SELECT Table_name, is_updatable
FROM information_schema.views
WHERE table_schema = 'MovieIndustry';

DELETE FROM ActorView
WHERE Id = 11;

-- WITH CHECK OPTION
CREATE OR REPLACE VIEW SingleActors AS 
SELECT FirstName, SecondName, DoB, Gender, MaritalStatus, NetWorthInMillions 
FROM Actors 
WHERE MaritalStatus = 'Single' 
WITH CHECK OPTION;

INSERT INTO SingleActors (FirstName, SecondName, DoB, Gender, MaritalStatus, NetWorthInMillions) 
VALUES ('Charlize', 'Theron', '1975-08-07', 'Female', 'Single', 130);

-- CHECK
CREATE OR REPLACE VIEW ActorsView2 AS
SELECT * 
FROM ActorsView1
WITH CASCADED CHECK OPTION; 

ALTER VIEW ActorsView2 AS
SELECT * 
FROM ActorsView1
WITH LOCAL CHECK OPTION; 

-- Drop, Show, & Rename Views
SHOW FULL TABLES
WHERE table_type = 'VIEW';

SELECT table_name
FROM information_schema.TABLES
WHERE table_type = 'VIEW'
AND table_schema = 'MovieIndustry';

DROP VIEW IF EXISTS DigitalAssetCount, ActorAssets;

CREATE VIEW ActorAge AS
SELECT * 
FROM Actors 
WHERE TIMESTAMPDIFF(YEAR, DoB, CURDATE()) > 50; 

-- Stored procedure
DELIMITER **
CREATE PROCEDURE ShowActors()
BEGIN
 SELECT *  FROM Actors;
END **
DELIMITER ;

CALL ShowActors();

SHOW PROCEDURE STATUS WHERE db = 
'MovieIndustry';

SELECT routine_name
FROM information_schema.routines
WHERE routine_type = 'PROCEDURE'
AND routine_schema = 'sys';

DROP PROCEDURE IF EXISTS ShowActors;

-- Variables
DECLARE TotalM, TotalF INT DEFAULT 0;

SET TotalM = 6;
SET TotalF = 4;

SELECT AVG(NetWorthInMillions)
INTO AvgNetWorth
FROM Actors;

DELIMITER **

CREATE PROCEDURE Summary()
BEGIN
 DECLARE TotalM, TotalF INT DEFAULT 0;
 DECLARE AvgNetWorth DEC(6,2) DEFAULT 0.0;
 
 SELECT COUNT(*) INTO TotalM
 FROM Actors
 WHERE Gender = 'Male';
 
 SELECT COUNT(*) INTO TotalF
 FROM Actors
 WHERE Gender = 'Female';
 
 SELECT AVG(NetWorthInMillions)
 INTO AvgNetWorth
 FROM Actors;
 
 SELECT TotalM, TotalF, AvgNetWorth;
END**

DELIMITER ;

CALL Summary();

-- Parameters
DELIMITER **

CREATE PROCEDURE GetActorCountByNetWorth (
 IN  NetWorth INT,
 OUT ActorCount INT
)
BEGIN
 SELECT COUNT(*)
 INTO ActorCount
 FROM Actors
 WHERE NetWorthInMillions >= NetWorth;
END**

DELIMITER ;

CALL GetActorCountByNetWorth(500, @ActorCount);
SELECT @ActorCount;

DELIMITER **

CREATE PROCEDURE IncreaseInNetWorth(
INOUT IncreasedWorth INT,
IN ActorId INT,
)
BEGIN
DECLARE OriginalNetWorth INT;

SELECT NetWorthInMillions Into OriginalNetWorth
FROM Actors WHERE Id = ActorId;

SET IncreasedWorth = OriginalNetWorth + IncreasedWorth;

END**
DELIMITER ;

SET @IncreasedWorth = 50;

CALL IncreaseInNetWorth(@IncreasedWorth, 11);

SELECT @IncreasedWorth;

DELIMITER **

CREATE PROCEDURE GenderCountByNetWroth(
 IN NetWorth INT,
 OUT MaleCount INT,
 OUT FemaleCount INT)
BEGIN
     SELECT COUNT(*) INTO MaleCount
     FROM Actors
     WHERE NetWorthInMillions >= NetWorth
           AND Gender = 'Male';

 SELECT COUNT(*) INTO FemaleCount
     FROM Actors
     WHERE NetWorthInMillions >= NetWorth
           AND Gender = 'Female';

END**
DELIMITER ;

CALL GenderCountByNetWroth(500, @Male, 
@Female);

SELECT @Male, @Female;

-- Conditional Statements
DELIMITER **

CREATE PROCEDURE GetMaritalStatus(
 IN  ActorId INT, 
 OUT ActorStatus  VARCHAR(30))
BEGIN
 DECLARE Status VARCHAR (15);

 SELECT MaritalStatus INTO Status
 FROM Actors
 WHERE Id = ActorId;

 IF Status LIKE 'Married' THEN
     SET ActorStatus = 'Actor is married';

 ELSEIF Status LIKE 'Single' THEN
     SET ActorStatus = 'Actor is single';

 ELSEIF Status LIKE 'Divorced' THEN
     SET ActorStatus = 'Actor is divorced';

 ELSE
     SET ActorStatus = 'Status not found';

 END IF;
END **

DELIMITER ;

CALL GetMaritalStatus(1, @status);
SELECT @status;

CALL GetMaritalStatus(5, @status);
SELECT @status;

CALL GetMaritalStatus(6, @status);
SELECT @status;

DROP PROCEDURE GetMaritalStatus;

DELIMITER **
CREATE PROCEDURE GetMaritalStatus(
 IN  ActorId INT, 
 OUT ActorStatus VARCHAR(30))
BEGIN
 DECLARE Status VARCHAR (15);

 SELECT MaritalStatus INTO Status
 FROM Actors 
 WHERE Id = ActorId;

 CASE Status
     WHEN 'Married' THEN
         SET ActorStatus = 'Actor is married';
     WHEN 'Single' THEN
         SET ActorStatus = 'Actor is single';
     WHEN 'Divorced' THEN
         SET ActorStatus = 'Actor is divorced';
     ELSE
         SET ActorStatus = 'Status not found';
 END CASE;
END**

DELIMITER ;

CALL GetMaritalStatus(1, @status);
SELECT @status;

CALL GetMaritalStatus(5, @status);
SELECT @status;

CALL GetMaritalStatus(6, @status);
SELECT @status;

-- Iteration
-- This is just a very cumbersome way of creating a list of names shown only as an example of using the LOOP statement in stored procedures. The same can be accomplished with a SELECT query by using Gender in the WHERE clause

DELIMITER **
CREATE PROCEDURE PrintMaleActors(
    OUT str  VARCHAR(255))
BEGIN

DECLARE TotalRows INT DEFAULT 0;
DECLARE CurrentRow INT;
DECLARE fname VARCHAR (25);
DECLARE lname VARCHAR (25);
DECLARE gen VARCHAR (10);

SET CurrentRow = 1;
SET str =  '';

SELECT COUNT(*) INTO TotalRows 
FROM Actors;

Print_loop: LOOP
 IF CurrentRow > TotalRows THEN
   LEAVE Print_loop;
 END IF;

SELECT Gender INTO gen 
FROM Actors 
WHERE Id = CurrentRow;

IF gen NOT LIKE 'Male' THEN
 SET CurrentRow = CurrentRow + 1;
 ITERATE Print_loop;
ELSE
 SELECT FirstName INTO fname 
 FROM Actors 
 WHERE Id = CurrentRow;

 SELECT SecondName INTO lname 
 FROM Actors 
 WHERE Id = CurrentRow;

 SET  str = CONCAT(str,fname,' ',lname,', ');
 SET CurrentRow = CurrentRow + 1;
END IF;
END LOOP Print_loop;

End **

DELIMITER ;

CALL PrintMaleActors(@namestr);
SELECT @namestr AS MaleActors;


DROP PROCEDURE PrintMaleActors;

DELIMITER **

CREATE PROCEDURE PrintMaleActors(
    OUT str  VARCHAR(255))
BEGIN

DECLARE TotalRows INT DEFAULT 0;
DECLARE CurrentRow INT;
DECLARE fname VARCHAR (25);
DECLARE lname VARCHAR (25);
DECLARE gen VARCHAR (10);

SET CurrentRow = 1;
SET str =  '';

SELECT COUNT(*) INTO TotalRows 
FROM Actors;

Print_loop: WHILE CurrentRow < TotalRows DO
 SELECT Gender INTO gen 
 FROM Actors 
 WHERE Id = CurrentRow;

 IF gen LIKE 'Male' THEN
   SELECT FirstName INTO fname 
   FROM Actors 
   WHERE Id = CurrentRow;

   SELECT SecondName INTO lname 
   FROM Actors 
   WHERE Id = CurrentRow;

   SET  str = CONCAT(str,fname,' ',lname,', ');
 END IF;
 
 SET CurrentRow = CurrentRow + 1;
END WHILE Print_loop;
End **

DELIMITER ;


DROP PROCEDURE PrintMaleActors;

DELIMITER **

CREATE PROCEDURE PrintMaleActors(
    OUT str  VARCHAR(255))
BEGIN

DECLARE TotalRows INT DEFAULT 0;
DECLARE CurrentRow INT;
DECLARE fname VARCHAR (25);
DECLARE lname VARCHAR (25);
DECLARE gen VARCHAR (10);

SET CurrentRow = 1;
SET str =  '';

SELECT COUNT(*) INTO TotalRows 
FROM Actors;

Print_loop: REPEAT 
 SELECT Gender INTO gen 
 FROM Actors 
 WHERE Id = CurrentRow;

 IF gen LIKE 'Male' THEN
   SELECT FirstName INTO fname 
   FROM Actors 
   WHERE Id = CurrentRow;

   SELECT SecondName INTO lname 
   FROM Actors 
   WHERE Id = CurrentRow;

   SET  str = CONCAT(str,fname,' ',lname,', ');
 END IF;
 
 SET CurrentRow = CurrentRow + 1;
 UNTIL CurrentRow > TotalRows
 END REPEAT Print_loop;

End **

DELIMITER ;

-- Cursor
DELIMITER **

CREATE PROCEDURE PrintMaleActors(
    OUT str  VARCHAR(255))
BEGIN

DECLARE fName VARCHAR(25);
DECLARE lName VARCHAR(25);
DECLARE LastRowFetched INTEGER DEFAULT 0;

DEClARE Cur_MaleActors CURSOR FOR 
 SELECT FirstName, SecondName 
 FROM Actors 
 WHERE Gender = 'Male';

DECLARE CONTINUE HANDLER FOR NOT FOUND 
 SET LastRowFetched = 1;

SET str =  '';

OPEN Cur_MaleActors;

Print_loop: LOOP
 FETCH Cur_MaleActors INTO fName, lName;

 IF LastRowFetched = 1 THEN
   LEAVE Print_loop;
 END IF;

 SET  str = CONCAT(str,fName,' ',lName,', ');
END LOOP Print_loop;

CLOSE Cur_MaleActors;
SET LastRowFetched = 0;

END **
DELIMITER ;

CALL PrintMaleActors(@namestr);
SELECT @namestr AS MaleActors;

-- Error Handling
DELIMITER **
CREATE PROCEDURE InsertDigitalAssets(
 IN Id INT,
 IN Asset VARCHAR (100),
 IN Type VARCHAR (25))
BEGIN
 DECLARE CONTINUE HANDLER FOR 1062
 BEGIN
 SELECT 'Duplicate key error occurred' AS Message;
 END;

 INSERT INTO DigitalAssets(URL, AssetType, ActorID) VALUES(Asset, Type, Id);

 SELECT COUNT(*) AS AssetsOfActor
 FROM DigitalAssets
 WHERE ActorId = Id;
END**
DELIMITER ;

CALL InsertDigitalAssets(10, 'https://instagram.com/iamsrk','Instagram');

DECLARE WrongCursorStatement CONDITION for 1322 ; 

DECLARE EXIT HANDLER FOR WrongCursorStatement 
 SELECT 'Provide a SELECT statement for the cursor' Message; 

 --- SIGNAL and RESIGNAL
 DELIMITER **

CREATE PROCEDURE AddActor(
              IN Name1 VARCHAR(20),
              IN Name2 VARCHAR(20), 
              IN Birthday DATE,
              IN networth INT )
BEGIN
 DECLARE age INT DEFAULT 0;
 SELECT TIMESTAMPDIFF (YEAR, Birthday, CURDATE())
 INTO age;

 IF(age < 1) THEN 
     SIGNAL SQLSTATE '45000'
     SET MESSAGE_TEXT = 'Incorrect DoB value. Age cannot be zero or less than zero';
 END IF;
 
 IF(networth < 1) THEN 
     SIGNAL SQLSTATE '45000'
         SET MESSAGE_TEXT = 'Incorrect NetWorth value. Net worth cannot be zero or less than zero';
 END IF;
 
 -- If all ok then INSERT a row in the Actors table
 INSERT INTO Actors (FirstName, SecondName, Dob, NetWorthInMillions) 
 VALUES(Name1, Name2, Birthday, networth);

END **
DELIMITER ;

CALL AddActor('Jackson','Samuel','2020-12-21', 250);

DROP PROCEDURE AddActor;

DELIMITER **
CREATE PROCEDURE AddActor(
           IN Name1 VARCHAR(20),
           IN Name2 VARCHAR(20), 
           IN Birthday DATE,
           IN networth INT)
BEGIN
 DECLARE age INT DEFAULT 0;
 DECLARE InvalidValue CONDITION FOR SQLSTATE '45000';

 DECLARE CONTINUE HANDLER FOR InvalidValue 
 IF age < 1 THEN
     RESIGNAL;
 ELSEIF networth < 1 THEN
     RESIGNAL;
 END IF;

 SELECT TIMESTAMPDIFF (YEAR, Birthday, CURDATE())
 INTO age;
   
 IF age < 1 THEN
     SIGNAL InvalidValue;
 ELSEIF networth < 1 THEN
     SIGNAL InvalidValue;
 ELSE
     INSERT INTO Actors (FirstName, SecondName, Dob, NetWorthInMillions) 
     VALUES(Name1, Name2, Birthday, networth);
 END IF;
END **

DELIMITER ;

DROP PROCEDURE AddActor;

DELIMITER **
CREATE PROCEDURE AddActor(
         IN FirstName varchar(20),
         IN SecondName varchar(20), 
         IN DoB date,
         IN networth int )
BEGIN
 DECLARE age INT DEFAULT 0;
 DECLARE InvalidValue CONDITION FOR SQLSTATE '45000';

 DECLARE CONTINUE HANDLER FOR InvalidValue
     IF age < 1 THEN 
         RESIGNAL SET MESSAGE_TEXT = 'Incorrect DoB value. Age cannot be zero or less than zero';
     ELSEIF networth < 1 THEN 
         RESIGNAL SET MESSAGE_TEXT = 'Incorrect NetWorth value. Net worth cannot be zero or less than zero';
     END IF;
 
 SELECT TIMESTAMPDIFF (YEAR, DoB, CURDATE())
 INTO age;

 IF age < 1 THEN 
     SIGNAL InvalidValue;
 ELSEIF networth < 1 THEN 
     SIGNAL InvalidValue;
 ELSE
     INSERT INTO Actors (FirstName, SecondName, Dob, NetWorthInMillions) 
     VALUES(Name1, Name2, Birthday, networth);
 END IF;
 
END **
DELIMITER ;

--- Stored Functions
DELIMITER **

CREATE FUNCTION DigitalAssetCount(
 ID INT) 
RETURNS VARCHAR(50)
DETERMINISTIC
BEGIN
 DECLARE ReturnMessage VARCHAR(50);
 DECLARE Number INT DEFAULT 0;
 SELECT COUNT(*) INTO Number FROM DigitalAssets WHERE ActorId = ID;

 IF Number = 0 THEN
     SET ReturnMessage = 'The Actor does not have any digital assets.';
 ELSE
     SET ReturnMessage = CONCAT('The Actor has ', Number, ' digital assets');
 END IF;
 
 -- return the customer level
 RETURN (ReturnMessage);
END**
DELIMITER ;

SHOW FUNCTION STATUS;

SELECT Id, DigitalAssetCount(Id) AS Count
FROM Actors;

DELIMITER **

CREATE PROCEDURE GetDigitalAssetCount(
 IN  ActorNo INT,  
 OUT Message VARCHAR(50))
BEGIN
 DECLARE Number INT DEFAULT 0;
 SET Number = ActorNo;    
 SET Message = DigitalAssetCount(Number);
END**

DELIMITER ;

DELIMITER **

CREATE FUNCTION TimeSinceLastUpdate(
             ID INT,
             Asset VARCHAR(15)) 
RETURNS INT
NOT DETERMINISTIC
BEGIN
 DECLARE ElapsedTime INT;

 SELECT TIMESTAMPDIFF(SECOND, LastUpdatedOn, NOW())
 INTO ElapsedTime
 FROM DigitalAssets
 WHERE ActorID = ID AND AssetType = Asset;

 RETURN ElapsedTime;
END**
DELIMITER ;

DROP FUNCTION DigitalAssetCount;
DROP FUNCTION IF EXISTS 
DigitalAssetCount;
SHOW WARNINGS;

--- Triggers
DELIMITER **
CREATE TRIGGER NetWorthCheck
BEFORE INSERT ON Actors
FOR EACH ROW 
 IF  NEW.NetWorthInMillions < 0 OR NEW.NetWorthInMillions IS NULL
THEN SET New.NetWorthInMillions = 0;
END IF;
**
DELIMITER ;

SHOW TRIGGERS;

INSERT INTO Actors (FirstName, SecondName, DoB, Gender, MaritalStatus,  NetWorthInMillions) VALUES ('Tom', 'Hanks', '1956-07-09', 'Male', 'Married', 350);

DROP TRIGGER IF EXISTS NetWorthCheck;

--- INSERT Trigger
DELIMITER **

CREATE TRIGGER BeforeActorsInsert
BEFORE INSERT ON Actors 
FOR EACH ROW
BEGIN
DECLARE TotalWorth, RowsCount INT;
       
SELECT SUM(NetWorthInMillions) INTO TotalWorth
FROM Actors;

SELECT COUNT(*) INTO RowsCount
FROM Actors;

UPDATE NetWorthStats
SET AverageNetWorth = ((Totalworth + new.NetWorthInMillions) / (RowsCount+1));
  
END **

DELIMITER ;

CREATE TRIGGER AfterActorsInsert 
AFTER INSERT ON Actors
FOR EACH ROW 
INSERT INTO ActorsLog
SET ActorId = NEW.Id, 
 FirstName = New.FirstName, 
 LastName = NEW.SecondName, 
 DateTime = NOW(), 
 Event = 'INSERT';

--- UPDATE Trigger
DELIMITER **

CREATE TRIGGER BeforeDigitalAssetUpdate
BEFORE UPDATE
ON DigitalAssets 
FOR EACH ROW
BEGIN
DECLARE errorMessage VARCHAR(255);

IF NEW.LastUpdatedOn < OLD.LastUpdatedOn THEN
SET errorMessage = CONCAT('The new value of LastUpatedOn column: ', 
  NEW.LastUpdatedOn,' cannot be less than the current value: ', 
  OLD.LastUpdatedOn);

SIGNAL SQLSTATE '45000'
SET MESSAGE_TEXT = errorMessage;
END IF;

IF NEW.LastUpdatedOn != OLD.LastUpdatedOn THEN
INSERT into DigitalActivity (ActorId, Detail)
VALUES (New.ActorId, CONCAT('LastUpdate value for ',NEW.AssetType,
       ' is modified from ',OLD.LastUpdatedOn, ' to ', 
       NEW.LastUpdatedOn));   
END IF;
  
END **

DELIMITER ;

DELIMITER **

CREATE TRIGGER AfterActorUpdate
AFTER UPDATE ON Actors 
FOR EACH ROW
BEGIN
DECLARE TotalWorth, RowsCount INT;

INSERT INTO ActorsLog
SET ActorId = NEW.Id, FirstName = New.FirstName, LastName =  NEW.SecondName, DateTime = NOW(), Event = 'UPDATE';

IF NEW.NetWorthInMillions != OLD.NetWorthInMillions THEN
 
SELECT SUM(NetWorthInMillions) INTO TotalWorth
 FROM Actors;

 SELECT COUNT(*) INTO RowsCount
 FROM Actors;

 UPDATE NetWorthStats
 SET AverageNetWorth = ((Totalworth) / (RowsCount));
END IF;
END **

DELIMITER ;

--- DELETE Trigger
DELIMITER **

CREATE TRIGGER BeforeActorsDelete
BEFORE DELETE
ON Actors
FOR EACH ROW
BEGIN
 INSERT INTO ActorsArchive 
      (Id, Firstname, SecondName, DoB, Gender, MaritalStatus, NetWorthInMillions)
VALUES (OLD.Id, OLD.Firstname, OLD.SecondName, OLD.DoB, OLD.Gender, OLD.MaritalStatus, OLD.NetWorthInMillions);
END **

DELIMITER ;

DELIMITER **

CREATE TRIGGER AfterActorsDelete
AFTER DELETE ON Actors 
FOR EACH ROW
BEGIN
DECLARE TotalWorth, RowsCount INT;

INSERT INTO ActorsLog
SET ActorId = OLD.Id, FirstName = OLD.FirstName, LastName =  OLD.SecondName, DateTime = NOW(), Event = 'DELETE';
 
SELECT SUM(NetWorthInMillions) INTO TotalWorth
FROM Actors;

SELECT COUNT(*) INTO RowsCount
FROM Actors;

UPDATE NetWorthStats
SET AverageNetWorth = ((Totalworth) / (RowsCount));
END **

DELIMITER ;

--- Multiple Triggers
DELIMITER **

CREATE TRIGGER UpdateGenderSummary
AFTER INSERT
ON Actors 
FOR EACH ROW
BEGIN

DECLARE count INT;

IF NEW.Gender = 'Male' THEN
 UPDATE GenderSummary
SET TotalMales = TotalMales+1;

INSERT INTO ActorsTableLog (ActorId, Detail) 
VALUES (NEW.Id, 'TotalMales value of GenderSummary table changed.');

ELSE  
UPDATE GenderSummary
SET TotalFemales = TotalFemales+1;

INSERT INTO ActorsTableLog (ActorId, Detail) 
VALUES (NEW.Id, 'TotalFemales value of GenderSummary table changed.');

END IF;
END  **

DELIMITER ;

DELIMITER **

CREATE TRIGGER UpdateMaritalStatusSummary
AFTER INSERT
ON Actors 
FOR EACH ROW
FOLLOWS UpdateGenderSummary

BEGIN

DECLARE count INT;

IF NEW.MaritalStatus = 'Single' THEN
 UPDATE MaritalStatusSummary
SET TotalSingle = TotalSingle+1;

INSERT INTO ActorsTableLog (ActorId, Detail) 
VALUES (NEW.Id, 'TotalSingle value of MaritalStatusSummary table changed.');

ELSEIF  NEW.MaritalStatus = 'Married' THEN
 UPDATE MaritalStatusSummary
SET TotalMarried = TotalMarried+1;

INSERT INTO ActorsTableLog (ActorId, Detail) 
VALUES (NEW.Id, 'TotalMarried value of MaritalStatusSummary table changed.');

ELSE
UPDATE MaritalStatusSummary
SET TotalDivorced = TotalDivorced+1;

INSERT INTO ActorsTableLog (ActorId, Detail) 
VALUES (NEW.Id, 'TotalDivorced value of MaritalStatusSummary table changed.');

END IF;
END  **

DELIMITER ;

-- Transaction

START TRANSACTION;

UPDATE Actors 
SET Id = 100 
WHERE FirstName = "Brad";

COMMIT;

START TRANSACTION;

UPDATE Actors 
SET Id = 200 
WHERE FirstName = "Tom";

ROLLBACK;

-- EXPLAIN

EXPLAIN SELECT * FROM Actors;

-- Foreign Keys
ALTER TABLE DigitalAssets
ADD FOREIGN KEY (ActorId)
REFERENCES Actors(Id);

--SET 1
-- The lesson queries are reproduced below for convenient copy/paste into the terminal. 

-- Question # 1, Query 1
SELECT Name 
FROM Movies 
ORDER BY CollectionInMillions DESC 
LIMIT 3;

-- Question # 2, Query 1
SELECT * FROM Movies a 
INNER JOIN Movies b;

-- Question # 2, Query 2
SELECT concat(a.FirstName," ",b.SecondName) 
FROM Actors a 
INNER JOIN Actors b 
ON a.SecondName = b.SecondName 
WHERE a.ID != b.ID;

-- Question # 2, Query 3
SELECT DISTINCT concat(a.FirstName," ",b.SecondName) 
AS Actors_With_Shared_SecondNames
FROM Actors a 
INNER JOIN Actors b 
ON a.SecondName = b.SecondName 
WHERE a.Id != b.Id;

-- Question # 3, Query 1
SELECT b.SecondName 
FROM Actors a 
INNER JOIN Actors b 
ON a.SecondName = b.SecondName 
WHERE a.Id != b.Id
GROUP BY b.SecondName;

-- Question # 3, Query 2
SELECT a.SecondName, 
COUNT(DISTINCT a.FirstName) 
FROM Actors a 
INNER JOIN Actors b 
ON a.SecondName = b.SecondName 
WHERE a.Id != b.Id 
group by a.SecondName;

-- Question # 3, Query 3
SELECT a.SecondName AS Actors_With_Shared_SecondNames, 
COUNT(DISTINCT a.Id) AS Count
FROM Actors a 
INNER JOIN Actors b 
ON a.SecondName = b.SecondName 
WHERE a.Id != b.Id 
group by a.SecondName;

-- Question # 4, Query 1
SELECT DISTINCT CONCAT(FirstName, " ", SecondName) AS Actors_Acted_In_Atleast_1_Movies 
FROM Actors
INNER JOIN Cast
ON Id = ActorId;

-- Question # 5, Query 1
SELECT Id, CONCAT(FirstName, " ", SecondName) AS Actors_With_No_Movies
FROM Actors
WHERE Id NOT IN (SELECT Id 
                 FROM Actors
                 INNER JOIN Cast
                 ON Id = ActorId);

-- Question # 5, Query 2
SELECT * 
FROM Actors
LEFT JOIN Cast
ON Id = ActorId;

-- Question # 5, Query 3
SELECT CONCAT(FirstName, " ", SecondName)  AS Actors_With_No_Movies 
FROM Actors
LEFT JOIN Cast
ON Id = ActorId
WHERE MovieId IS NULL;

-- Practice Set 2
-- The lesson queries are reproduced below for convenient copy/paste into the terminal. 

-- Question # 1, Query 1
SELECT Id, FirstName, SecondName, MovieId
FROM Actors 
INNER JOIN Cast
ON Id = ActorId;

-- Question # 1, Query 2
SELECT Id, COUNT(*)
FROM Actors 
INNER JOIN Cast
ON Id = ActorId
GROUP BY Id;

-- Question # 1, Query 3
SELECT Id, 
COUNT(*) AS MovieCount
FROM Actors 
INNER JOIN Cast
ON Id = ActorId
GROUP BY Id
HAVING MovieCount > 1;

-- Question # 1, Query 4
SELECT CONCAT (FirstName, " ", SecondName)
AS Actor_Names, 
Movie_Count 
FROM Actors a
INNER JOIN (SELECT Id, 
            COUNT(*) AS Movie_Count
            FROM Actors 
            INNER JOIN Cast
            ON Id = ActorId
            GROUP BY Id
            HAVING Movie_Count > 1) AS tbl
ON tbl.Id = a.Id;

-- Question # 2, Query 1
SELECT Id 
FROM Movies 
WHERE Name="Mr & Mrs. Smith";

-- Question # 2, Query 2
SELECT ActorId
FROM Cast
WHERE MovieId IN (SELECT Id 
                  FROM Movies 
                  WHERE Name="Mr & Mrs. Smith");

-- Question # 2, Query 3
SELECT CONCAT(FirstName, " ", SecondName) 
AS "Cast Of Mr. & Mrs. Smith"
FROM Actors
WHERE Id IN ( SELECT ActorId
              FROM Cast
              WHERE MovieId IN (SELECT Id 
                                FROM Movies 
                                WHERE Name="Mr & Mrs. Smith"));

-- Question # 2, Query 4
SELECT CONCAT(FirstName, " ", SecondName) 
AS "Cast Of Mr. & Mrs. Smith"
FROM Actors
INNER JOIN (SELECT ActorId
            FROM Cast
            INNER JOIN Movies
            ON MovieId = Id
            WHERE Name="Mr & Mrs. Smith") AS tbl
ON tbl.ActorId = Id;

-- Question # 3, Query 1
SELECT Name, ActorId
FROM Movies
INNER JOIN Cast
On Id = MovieId; 

-- Question # 3, Query 2
SELECT tbl.Name AS Movie_Name,
CONCAT(FirstName, " ", SecondName) AS Actor_Name
FROM Actors
INNER JOIN (SELECT Name, ActorId
            FROM Movies
            INNER JOIN Cast
            On Id = MovieId) AS tbl 
ON tbl.ActorId = Id
ORDER BY tbl.Name ASC;

-- Question # 4, Query 1
SELECT tbl.Name, COUNT(*)
FROM Actors
INNER JOIN (SELECT Name, ActorId, MovieId
            FROM Movies
            INNER JOIN Cast
            On Id = MovieId) AS tbl 
ON tbl.ActorId = Id
GROUP BY tbl.MovieId;

-- Question # 4, Query 2
SELECT MovieId, COUNT(*)
FROM Cast
GROUP BY MovieId;

-- Question # 4, Query 3
SELECT Name AS Movie_Name, 
Actor_Count
FROM Movies
INNER JOIN (SELECT MovieId, COUNT(*) AS Actor_Count
FROM Cast
GROUP BY MovieId) AS tbl
ON tbl.MovieID = Id;

-- Question # 5, Query 1
SELECT Id
FROM Actors 
WHERE FirstName = "Tom"
AND SecondName = "Cruise";

-- Question # 5, Query 2
SELECT MovieId
FROM Cast
WHERE ActorId = (SELECT Id
                  FROM Actors 
                  WHERE FirstName = "Tom"
                  AND SecondName = "Cruise");

-- Question # 5, Query 3
SELECT DISTINCT Producer 
FROM Movies 
WHERE Id IN (SELECT MovieId
             FROM Cast
             WHERE ActorId = (SELECT Id
                               FROM Actors 
                               WHERE FirstName = "Tom"
                               AND SecondName = "Cruise"));

-- Question # 5, Query 4
SELECT DISTINCT Producer 
FROM Movies 
WHERE Id NOT IN (SELECT MovieId
             FROM Cast
             WHERE ActorId = (SELECT Id
                               FROM Actors 
                               WHERE FirstName = "Tom"
                               AND SecondName = "Cruise"));

-- Question # 5, Query 5
SELECT DISTINCT Producer
FROM Movies
WHERE Producer
NOT IN (SELECT Producer 
        FROM Movies 
        WHERE Id IN (SELECT MovieId
                     FROM Cast
                     WHERE ActorId = (SELECT Id
                                      FROM Actors 
                                      WHERE FirstName = "Tom"
                               AND SecondName = "Cruise")));
                               
-- Practice Set 3
-- The lesson queries are reproduced below for convenient copy/paste into the terminal. 

-- Question # 1, Query 1
SELECT AVG(BudgetInMillions) 
FROM Movies;

-- Question # 1, Query 2
SELECT Name 
FROM Movies 
WHERE BudgetInMillions > (SELECT AVG(BudgetInMillions) 
                          FROM Movies);

-- Question # 2, Query 1
SELECT * FROM DigitalAssets 
RIGHT JOIN Actors 
ON Id = ActorId;

-- Question # 2, Query 2
SELECT CONCAT(FirstName, " ", SecondName)
AS Actors_With_No_Online_Presence
FROM DigitalAssets 
RIGHT JOIN Actors 
ON Id = ActorId
WHERE URL IS NULL;

-- Question # 3, Query 1
SELECT CONCAT(FirstName, " ", SecondName)
FROM Actors 
WHERE NOT EXISTS (SELECT ActorId 
                  FROM DigitalAssets 
                  WHERE ActorId = Id);

-- Question # 4, Query 1
SELECT Name, CollectionInMillions
FROM Movies
ORDER BY CollectionInMillions DESC;

-- Question # 4, Query 2
SELECT Name, 
CollectionInMillions AS Collection_In_Millions
FROM Movies
ORDER BY CollectionInMillions DESC
LIMIT 1 OFFSET 4;

-- Question # 4, Query 3
SELECT Name, 
CollectionInMillions AS Collection_In_Millions
FROM Movies
ORDER BY CollectionInMillions DESC
LIMIT 4, 1;

-- Question # 5, Query 1
SELECT LastUpdatedOn, Id 
FROM Actors 
INNER JOIN DigitalAssets 
ON ActorId = Id;

-- Question # 5, Query 2
SELECT * 
FROM Cast 
INNER JOIN (SELECT LastUpdatedOn, Id 
            FROM Actors 
            INNER JOIN DigitalAssets 
            ON ActorId = Id) AS tbl 
ON tbl.Id = ActorId;

-- Question # 5, Query 3
SELECT * 
FROM Movies AS m 
INNER JOIN (SELECT * 
            FROM Cast 
            INNER JOIN (SELECT LastUpdatedOn, Id 
                        FROM Actors 
                        INNER JOIN DigitalAssets 
                        ON ActorId = Id) AS tbl1
            ON tbl1.Id = ActorId) AS tbl2
ON tbl2.MovieId = m.Id;

-- Question # 5, Query 4
SELECT DISTINCT Name 
AS Actors_Posting_Online_Within_Five_Days_Of_Movie_Release
FROM Movies AS m 
INNER JOIN (SELECT * 
            FROM Cast 
            INNER JOIN (SELECT LastUpdatedOn, Id 
                        FROM Actors 
                        INNER JOIN DigitalAssets 
                        ON ActorId = Id) AS tbl1
            ON tbl1.Id = ActorId) AS tbl2
ON tbl2.MovieId = m.Id
WHERE ADDDATE(ReleaseDate, INTERVAL -5 Day) <= LastUpdatedOn
AND ADDDATE(ReleaseDate, INTERVAL +5 Day) >= LastUpdatedOn;

--Practice Set 4
-- The lesson queries are reproduced below for convenient copy/paste into the terminal. 

-- Question # 1, Query 1
SELECT Producer 
FROM Movies
GROUP BY Producer;

-- Question # 1, Query 2
SELECT Producer 
FROM Movies
GROUP BY Producer
HAVING COUNT(Producer) > 1;

-- Question # 1, Query 3
SELECT Producer AS Producer_Name, AVG(CollectionInMillions) AS Average_Collection_In_Millions
FROM Movies
GROUP BY Producer
HAVING COUNT(Producer) > 1;

-- Question # 2, Query 1
SELECT CONCAT (FirstName, " ", SecondName) AS Actors, MovieId, Producer 
FROM Actors JOIN Cast 
ON Actors.Id = Cast.ActorId
JOIN Movies
ON Cast.MovieId = Movies.Id;

-- Question # 2, Query 2
SELECT CONCAT (FirstName, " ", SecondName) AS Actors, MovieId, Producer 
FROM Actors JOIN Cast 
ON Actors.Id = Cast.ActorId
JOIN Movies
ON Cast.MovieId = Movies.Id
AND Producer <> 'Ryan Seacrest';

-- Question # 2, Query 3
SELECT DISTINCT(CONCAT (FirstName, " ", SecondName)) AS Actors_Who_Have_Not_Worked_with_Ryan_Seacrest 
FROM Actors  JOIN Cast 
ON Actors.Id = Cast.ActorId
JOIN Movies
ON Cast.MovieId = Movies.Id
AND Producer <> 'Ryan Seacrest';

-- Question # 2, Query 4
SELECT c.ActorID, c.MovieId, m.Producer 
FROM Cast c, Movies m 
WHERE c.MovieId = m.Id;

-- Question # 2, Query 5
SELECT c.ActorID, c.MovieId, m.Producer 
FROM Cast c, Movies m 
WHERE c.MovieId = m.Id 
  AND m.Producer <> 'Ryan Seacrest';

-- Question # 2, Query 6
SELECT DISTINCT(CONCAT (a.FirstName, " ", a.SecondName)) AS Actors_Who_Have_Not_Worked_with_Ryan_Seacrest 
FROM Cast c, Movies m, Actors a 
WHERE c.MovieId = m.Id 
  AND m.Producer <> 'Ryan Seacrest' 
  AND c.ActorId = a.Id;

-- Question # 3, Query 1
SELECT ActorId, AssetType, LastUpdatedOn 
FROM DigitalAssets 
ORDER BY ActorId ASC, 
         LastUpdatedOn DESC;

-- Question # 3, Query 2
SELECT ActorId, MAX(LastUpdatedOn)
FROM DigitalAssets 
GROUP BY ActorId;

-- Question # 3, Query 3
SELECT ActorId, AssetType, LastUpdatedOn 
FROM DigitalAssets
WHERE (ActorId, LastUpdatedOn) IN
                                (SELECT ActorId, MAX(LastUpdatedOn) 
                                 FROM DigitalAssets 
                                 GROUP BY ActorID);

-- Question # 3, Query 4
CREATE TABLE DigitalActivityTrack (
Id INT AUTO_INCREMENT NOT NULL PRIMARY KEY,
Actor_Id INT NOT NULL,
Digital_Asset VARCHAR(20) NOT NULL,
Last_Updated_At DATETIME Not NULL DEFAULT NOW()
);

-- Question # 3, Query 5
INSERT INTO DigitalActivityTrack (Actor_Id, Digital_Asset, Last_Updated_At)
SELECT ActorId, AssetType, LastUpdatedOn FROM DigitalAssets
                WHERE (ActorId, LastUpdatedOn) In 
                                (SELECT ActorId, MAX(LastUpdatedOn) FROM DigitalAssets 
                                 GROUP BY ActorID)
             ORDER BY LastUpdatedOn DESC;

-- Question # 3, Query 6
SELECT CONCAT(a.FirstName, " ", a.SecondName) AS Actor_Name, Digital_Asset, Last_Updated_At
FROM Actors a, DigitalActivityTrack
WHERE a.Id = Actor_Id;

-- Question # 4, Query 1
SELECT CONCAT (FirstName, " ", SecondName) AS Actor_Name, NetWorthInMillions AS 3rd_Lowest_Net_Worth_In_Millions
From Actors a1
WHERE 2 = (SELECT COUNT(DISTINCT (NetWorthInMillions)) 
           FROM Actors a2
           WHERE a2. NetWorthInMillions < a1. NetWorthInMillions);

-- Question # 5, Query 1
SELECT ActorID, COUNT(ActorId)
FROM DigitalAssets
GROUP BY ActorId;

-- Question # 5, Query 2
SELECT ActorID, GROUP_CONCAT(AssetType)
FROM DigitalAssets
GROUP BY ActorId;

-- Question # 5, Query 3
SELECT CONCAT (FirstName, " ", SecondName) AS Actor_Name,                
       GROUP_CONCAT(AssetType) AS Digital_Assets
FROM Actors INNER JOIN DigitalAssets
ON Actors.Id = DigitalAssets.ActorId
GROUP BY Id;

-- Question # 5, Query 4
SELECT CONCAT (FirstName, " ", SecondName) AS Actor_Name, 
       GROUP_CONCAT(AssetType) AS Digital_Assets
FROM Actors LEFT JOIN DigitalAssets
ON Actors.Id = DigitalAssets.ActorId
GROUP BY Id;

-- Practice Set 5
-- The lesson queries are reproduced below for convenient copy/paste into the terminal. 

-- Question # 1, Query 1
SELECT Weekend, RevenueInMillions
FROM MovieScreening
WHERE MovieId = 10
ORDER BY Weekend;

-- Question # 1, Query 2
SELECT T1.Weekend, T2.Weekend, T1.RevenueInMillions, T2.RevenueInMillions
FROM MovieScreening T1 
INNER JOIN MovieScreening T2
ON T1.MovieId = T2.MovieId
AND T1.Weekend >= T2.Weekend
WHERE T1.MovieId = 10
ORDER BY T1.Weekend, T1.RevenueInMillions;

-- Question # 1, Query 3
SELECT T1.Weekend, T1.RevenueInMillions, SUM(T2.RevenueInMillions) AS RunningTotal
FROM MovieScreening T1 
INNER JOIN MovieScreening T2
ON T1.MovieId = T2.MovieId
AND T1.Weekend >= T2.Weekend
WHERE T1.MovieId = 10
GROUP BY T1.Weekend, T1.RevenueInMillions
ORDER BY T1.Weekend, T1.RevenueInMillions;

-- Question # 2, Query 1
CREATE VIEW CollectionPerGenre AS
SELECT Genre, Sum(CollectionInMillions) as GenreTotal
From Movies
GROUP BY Genre
ORDER BY 2 DESC;

SELECT * FROM CollectionPerGenre;

-- Question # 2, Query 2
SELECT T1.Genre, T2.Genre, T1.GenreTotal, T2.GenreTotal
FROM CollectionPerGenre T1 
INNER JOIN CollectionPerGenre T2
WHERE T1.GenreTotal <= T2.GenreTotal
GROUP BY T1.Genre, T2.Genre, T1.GenreTotal
ORDER BY T1.GenreTotal DESC, T2.GenreTotal DESC;

-- Question # 2, Query 3
SELECT T1.Genre, T1.GenreTotal, 
       SUM(T2.GenreTotal) AS RunningTotal
FROM CollectionPerGenre T1 
INNER JOIN CollectionPerGenre T2
WHERE T1.GenreTotal <= T2.GenreTotal
GROUP BY T1.Genre, T1.GenreTotal
ORDER BY T1.GenreTotal DESC;

-- Question # 2, Query 4
SELECT  d1.Genre AS Genre,  
        d1.GenreTotal AS TotalRevenueInMillions,
        (d1.RunningTotal / d2.TotalSum) * 100 AS PercentageOfTotalRevenues
FROM
( SELECT T1.Genre as Genre, 
         T1.GenreTotal AS GenreTotal, 
         SUM(T2.GenreTotal) AS RunningTotal
  FROM CollectionPerGenre T1 
  INNER JOIN CollectionPerGenre T2
  WHERE T1.GenreTotal <= T2.GenreTotal
  GROUP BY T1.Genre, T1.GenreTotal
  ORDER BY T1.GenreTotal DESC ) d1,
( SELECT SUM(GenreTotal) AS TotalSum 
  FROM CollectionPerGenre) d2;

-- Question # 3, Query 1
SELECT   Weekend, RevenueInMillions
FROM  MovieScreening
WHERE  MovieId = 5
ORDER BY Weekend;

-- Question # 3, Query 2
SELECT @counter := @counter + 1 AS RowNum, 
       T1.Weekend, T1.RevenueInMillions
FROM MovieScreening T1, (SELECT @counter := 0) c
WHERE MovieId = 5
ORDER BY T1.Weekend;

-- Question # 3, Query 3
SELECT * 
FROM (
       SELECT @counter1 := @counter1+ 1 AS RowNum,
              T1.Weekend, T1.RevenueInMillions
       FROM MovieScreening T1, (SELECT @counter1 := 0) c1
       WHERE MovieId=5
       ORDER BY T1.Weekend ) AS table1
JOIN  
( SELECT @counter2 := @counter2 + 1 AS RowNum,
         T2.Weekend, T2.RevenueInMillions
  FROM MovieScreening T2, (SELECT @counter2 := 0) c2
  WHERE MovieId=5
  ORDER BY T2.Weekend ) AS table2

ON table2.RowNum <= table1.RowNum AND table2.RowNum > table1.RowNum - 3
ORDER BY table1.Weekend, table2.Weekend;

-- Question # 3, Query 4
SELECT table1.Weekend AS Weekend,
       table1.RevenueInMillions AS Revenue,
       SUM(table2.RevenueInMillions) AS 3WeekTotal,
       AVG(table2.RevenueInMillions) AS 3WeekAverage
FROM (
       SELECT @counter1 := @counter1+ 1 AS RowNum,
              T1.Weekend, T1.RevenueInMillions
       FROM MovieScreening T1, (SELECT @counter1 := 0) c1
       WHERE MovieId=5
       ORDER BY T1.Weekend ) AS table1
       JOIN  
       (SELECT @counter2 := @counter2 + 1 AS RowNum,
               T2.Weekend, T2.RevenueInMillions
        FROM MovieScreening T2, (SELECT @counter2 := 0) c2
        WHERE MovieId=5
        ORDER BY T2.Weekend ) AS table2
        ON table2.RowNum <= table1.RowNum AND table2.RowNum > table1.RowNum - 3
GROUP BY table1.RowNum, table1.Weekend, table1.RevenueInMillions
HAVING COUNT(table1.RowNum) > 2;

-- Question # 4, Query 1
SELECT Weekend, MONTH(Weekend) AS Month, ROUND(RevenueInMillions,2) As Revenue
FROM MovieScreening
WHERE MovieId = 2;

-- Question # 4, Query 2
SELECT MONTH(Weekend) As Month, group_concat(ROUND(RevenueInMillions,2))
FROM MovieScreening 
WHERE MovieId = 2
GROUP BY MONTH(Weekend);

-- Question # 4, Query 3
SELECT MONTH(Weekend) AS Month, 
       GROUP_CONCAT(ROUND(RevenueInMillions,2)) AS List,
       SUBSTRING_INDEX(GROUP_CONCAT(ROUND(RevenueInMillions,2)),  ',' , 1) AS FirstValue 
FROM MovieScreening 
WHERE MovieId = 2 
GROUP BY MONTH(Weekend);

-- Question # 4, Query 4
SELECT Weekend As Date, MONTH(Weekend) AS Month, 
       ROUND(RevenueInMillions,2) AS RevenueInMillions, FirstValue 
FROM MovieScreening t1,
     (SELECT MONTH(Weekend) AS Month, 
             SUBSTRING_INDEX(GROUP_CONCAT(ROUND(RevenueInMillions,2)),  ',' , 1) AS FirstValue 
      FROM MovieScreening 
      WHERE MovieId = 2 
      GROUP BY MONTH(Weekend) ) t2
WHERE t1.MovieId = 2 AND MONTH(t1.Weekend) = t2.Month;

-- Question # 4, Query 5
SELECT MONTH(Weekend) AS Month, 
       GROUP_CONCAT(ROUND(RevenueInMillions,2)) As List, 
       SUBSTRING_INDEX(GROUP_CONCAT(ROUND(RevenueInMillions,2)),  ',' , -1) AS LastValue 
FROM MovieScreening 
WHERE MovieId = 2 
GROUP BY MONTH(Weekend);

-- Question # 5, Query 1
SELECT MONTH(Weekend) AS Month, SUM(RevenueInMillions) AS TotalRevenueInMillions 
FROM MovieScreening 
WHERE MovieId = 3 
GROUP BY MONTH(Weekend);

-- Question # 5, Query 2
SELECT Month, TotalRevenueInMillions, 
       IF(@PrevVal = 0, 0, ROUND(((TotalRevenueInMillions - @PrevVal) / @PrevVal) * 100, 2))  "Growth %",
       @PrevVal := TotalRevenueInMillions         
FROM
      ( SELECT @PrevVal := 0) d1,
      ( SELECT MONTH(Weekend) AS Month, 
               SUM(RevenueInMillions) as TotalRevenueInMillions 
        FROM MovieScreening 
        WHERE MovieId = 3 
        GROUP BY MONTH(Weekend) ) d2;



