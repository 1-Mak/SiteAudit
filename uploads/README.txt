******************************************************************************
* (c) Copyright IBM Corp. 2007 All rights reserved.
* 
* The following sample of source code ("Sample") is owned by International 
* Business Machines Corporation or one of its subsidiaries ("IBM") and is 
* copyrighted and licensed, not sold. You may use, copy, modify, and 
* distribute the Sample in any form without payment to IBM, for the purpose of 
* assisting you in the development of your applications.
* 
* The Sample code is provided to you on an "AS IS" basis, without warranty of 
* any kind. IBM HEREBY EXPRESSLY DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR 
* IMPLIED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
* MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE. Some jurisdictions do 
* not allow for the exclusion or limitation of implied warranties, so the above 
* limitations or exclusions may not apply to you. IBM shall not be liable for 
* any damages you suffer as a result of using, copying, modifying or 
* distributing the Sample, even if IBM has been advised of the possibility of 
* such damages.
*
******************************************************************************
*
* README for Visual Basic ADO Samples on Windows
*
* The listed sample files are available in the
* <install_path>\sqllib\samples\VB\ADO directory.
* The default location for <install_path> is C:\Program Files\IBM.
*
* WARNING: Some of these samples may change your database or database manager
*          configuration. Execute the samples against a 'test' database only,
*          such as the DB2 SAMPLE database.
*
******************************************************************************
*
*               REQUIREMENTS
*
* 1. Visual Basic 6.0 Profession/Corporate Edition.
*
* 2. MDAC 2.8. Microsoft Data Access 2.8 SDK can be found at:
*
*      http://www.microsoft.com/data/download.htm
*
*    Microsoft Data Access Components (MDAC) enable Universal Data Access.
*    These components include Microsoft ActiveX Data Objects (ADO), OLE DB,
*    and Open Database Connectivity (ODBC).
*
* 3. MDAC 2.6 Latest Service Pack 1. You can download it from the Microsoft
*    Web Site as well. See above.
*
* 4. Visual Basic Latest Service pack from the Visual Studio Web Site.
*    Visual Studio Service Packs can be found at:
*
*      http://msdn.microsoft.com/vstudio/
*
******************************************************************************
*
*               Prepare your DB2 sample development environment
*
* 1) Copy the files in <install_path>\sqllib\samples\VB\ADO\* to a working
*    directory and ensure that directory has write permission.
*
*    All samples should be run and built in a DB2 Command Window.
*    The DB2 Command Window is needed to execute db2 specific commands.
*    You can follow the step below to open DB2 Command window.
*    From the Start Menu click Start --> Programs --> IBM DB2 -->
*    <DB2 copy name> --> Command Line Tools --> Command Window.
*
* 2) Start the Database Manager with the following command:
*      db2start
*
* 3) Create the sample database with the following command:
*      db2sampl
*
* 4) Connect to the database with the following command:
*      db2 connect to sample
*
* 5) To build Stored Procedures and User Defined Functions, ensure that you
*    have write permission on the <install_path>\sqllib\function directory.
*
* 6) cd to the directory containing the files copied in Step 1.
*
******************************************************************************
*
*               QUICKSTART
*
* Load and run the project file \Demo\Demo.vbp.
*
******************************************************************************
*
* This demo shows three different ways to connect to DB2. They are OLE DB,
* ODBC and DataShape. This demo contains five tabs, each of these shows a
* different way to work with the sample database. Some of the functions may
* only work with one or more connection method(s). See the table below for
* details.
*
* ____________________________________________________________________________
*| Connection |  Execute  | Hierarchical |           |   Stored   |           |
*|  methods   |    SQL    |     Data     |   LOBs    | Procedures |   UDFs    |
*|   \tabs    |           |              |           |            |           |
*|____________|___________|______________|___________|____________|___________|
*|            |           |              |           |            |           |
*|   OLE DB   | Available |      N/A     | Available | Available  | Available |
*|____________|___________|______________|___________|____________|___________|
*|            |           |              |           |            |           |
*|    ODBC    | Available |      N/A     | Available |    N/A     | Available |
*|____________|___________|______________|___________|____________|___________|
*|            |           |              |           |            |           |
*| DATA SHAPE | Available |  Available   | Available | Available  | Available |
*|____________|___________|______________|___________|____________|___________|
*
*
* STORED PROCEDURES:
* ------------------
* To run the stored procedure samples, you have to compile a stored procedure
* server sample program and then create and catalog the stored procedures in
* C, CLI, C++, JDBC or SQLj. For instance, if you choose to work with the
* stored procedures in C, do the following steps:
*
* 1. Compile the server source file spserver.sqc. Do the following in a DB2
*    CLP window:
*      nmake/make spserver
*
* 2. Call the script 'spcat' to create and catalog the Stored procedures.
*    Type the following to run the script:
*      spcat
*
* 3. At this point, some of the radio buttons should be enabled under the
*    "Stored procedures" tab. Click on any available radio button.
*
* 4. Then click on the "Call" button to call the stored procedure you have
*    choose.
*
* Note:
*   Not all of the radio buttons are enabled because some stored procedures
*   might not be available in a particular language. See the READMEs and the
*   header of the stored procedure sample programs in C, CLI, C++, JDBC or
*   SQLj for more information.
*
* UDFs:
* -----
* To run the UDFs, you have to compile an UDF server program udfsrv/UDFsrv
* in C, CLI, C++, JDBC or SQLj. For instance, if you choose to work with the
* UDFs in C, do the following steps:
*
* 1. Compile the server source file udfsrv.c. Do the following in a DB2
*    CLP window:
*      nmake/make udfsrv
*
* 2. You can then click on the buttons under the "UDFs" tab. Each of the
*    buttons will create a UDF and then execute an SQL statement that make
*    use of the UDF created. See the READMEs and the header of UDF sample
*    programs in C, CLI, C++, JDBC or SQLj for more information.
*
******************************************************************************
*
*              Common file Descriptions
* 
* The following are the common files for VB ADO samples. For more
* information on these files, refer to the program source files.
*
******************************************************************************
*
*  "ReadMe.txt"    - this file!
*
******************************************************************************
*
* Other Files
*
* Demo\Demo.frm  - Visual Basic file required for Demo.vbs
* Demo\Demo.frx  - Visual Basic file required for Demo.vbs
* Demo\Demo.vbw  - Visual Basic file required for Demo.vbs
*
******************************************************************************
*
*               VB Samples Design
*
* The Visual Basic ADO sample programs are organized to reflect an
* object-based design of the distinct levels of DB2. The level to which a
* sample belongs is indicated by a two character identifier at the beginning
* of the sample name. These levels show a hierarchical structure. A client
* application can access different databases, which hold data of different
* data types. Here are the DB2 levels demonstated by the Visual Basic ADO
* samples:
*
* Identifier     DB2 Level
*
*     cl        Client Level
*     db        Database Level
*     dt        Data Type Level
*
* Other Samples:
* Besides the samples organized in the DB2 Level design, other samples show
* specific kinds of application methods:
*
* Identifier     Application Method
*
*     sp        Stored Procedures
*     ud        User Defined Functions
*
******************************************************************************
*
*               VB ADO sample File Descriptions
*
* The following are the VB ADO sample files included with DB2. For more
* information on the sample programs, refer to the program source files.
*
******************************************************************************
*
* Client Level (deals with the client application level of DB2)
*
* "cliExeSQL.bas"  - How to execute SQL statements.
* "cli_Info.bas"   - How to get and set client level information.
*
******************************************************************************
*
* Database Level (deals with database objects in DB2)
*
* "dbConn.bas"    - How to connect and disconnect from a database.
* "dbInfo.bas"    - How to get and set information on database level.
* "dbCommit.bas"  - How to control autocommit dynamically on database level.
*
******************************************************************************
*
* Data Type Level (deals with data types).
*
* "dtHier.bas"    - How to retrieve hierarchical data
* "dtLob.bas"     - How to read and write LOB data.
*
******************************************************************************
*
* Stored Procedures (samples demonstrating stored procedures)
*
* "spCall.bas"    - How to call stored procedures.
*
******************************************************************************
*
* UDFs (samples demonstrating user defined functions)
*
* "udfUse.bas"    - How to create and work with UDTs and UDFs.
*
******************************************************************************
*
* Common Utility Function files
*
* "Util.bas"      - Common utilities for other sample programs
*
******************************************************************************
*
* Visual Basic Demo
*
* Demo\Demo.vbp - Visual Basic Demo with user interface for the sample modules
*
******************************************************************************
*
*               How to build your own application programs using ADO
*
* ActiveX Data Objects (ADO) allow you to write an application to access
* and manipulate data on a database server through an OLE DB provider. The
* primary benefits of ADO are high speed development time, ease of use, and
* a small disk footprint. You can write database applications that conform
* with ADO in Microsoft Visual Basic or Visual C++.
*
* To use ADO with Microsoft Visual Basic, you need to establish a
* reference to the ADO type library. Do the following:
*
* 1. Select "References" from the Project menu
* 2. Check the box for "Microsoft ActiveX Data Objects <version_number>
*    Library".
* 3. Click "OK".
*
* where <version_number> is the current version the ADO library.
*
* Once this is done, ADO objects, methods, and properties will be
* accessible through the VBA Object Browser and the IDE Editor.
*
* A full Visual Basic program includes forms and other graphical elements,
* and you need to view it inside the Visual Basic environment. Here are
* Visual Basic commands as part of a program to access the DB2 sample
* database, cataloged in ODBC:
*
* Establish a connection:
*
*   Dim con As ADODB.Connection
*   Set con = New ADODB.Connection
*
* Set client-side cursors supplied by the local cursor library:
*
*   con.CursorLocation = adUseClient
*
* Set the provider so ADO will use the IBM OLE DB provider, and open
* database "SAMPLE" with no user id/password; that is, use the current
* user:
*
*   con.Open "Provider=IBMDADB2;DSN=SAMPLE;User Id=;Password=;"
*
* Create a recordset object:
*
*   Dim rst As ADODB.Recordset
*   Set rst = New ADODB.Recordset
*
* Use a select statement to fill the record set:
*
*   rst.Open "SELECT * FROM EMPLOYEE", con
*
* From this point, the programmer can use the ADO methods to access the
* data such as moving to the next record set:
*
*   rst.MoveNext
*
* Deleting the current record in the record set:
*
*   rst.Delete
*
* As well, the programmer can do the following to access an individual
* field:
*
*   Dim Text1 as String
*   Text1 = rst!LASTNAME
*
******************************************************************************