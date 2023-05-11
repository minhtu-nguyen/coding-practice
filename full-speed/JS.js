// *** NODEJS
// --- How to create a simple server in Node.js
// npm install express --save
/**
 * Express.js
 */
const express = require('express');
// const app = express();

app.get('/', function (req, res) {
  res.send('Hello World!');
});

app.listen(3000, function () {
  console.log('App listening on port 3000!');
});

/**
 * URL Module in Node.js
 */
const url = require('url');
const adr = 'http://localhost:8080/default.htm?year=2022&month=september';
const q = url.parse(adr, true);

console.log(q.host); // localhost:8080
console.log(q.pathname); // "/default.htm"
console.log(q.search); // "?year=2022&month=september"

const qdata = q.query; // { year: 2022, month: 'september' }
console.log(qdata.month); // "september"

/**
 * BigInt Data Type
 */
const maxSafeInteger = 99n; // This is a BigInt
const num2 = BigInt('99'); // This is equivalent
const num3 = BigInt(99); // Also works

typeof 1n === 'bigint'           // true
typeof BigInt('1') === 'bigint'  // true

/**
 * Symbol Data Type -- CHECK THIS
 */
const NAME = Symbol()
const person = {
  [NAME]: 'Ritika Bhavsar'
}

person[NAME] // 'Ritika Bhavsar'

/**
 * Callback Events with Parameters
 */
const events = require('events');
// const eventEmitter = new events.EventEmitter();

function listener(code, msg) {
   console.log(`status ${code} and ${msg}`);
}

eventEmitter.on('status', listener); // Register listener
eventEmitter.emit('status', 200, 'ok');

// Output
// status 200 and ok

/**
 * Callbacks Events
 */
const events = require('events');
// const eventEmitter = new events.EventEmitter();

function listenerOne() {
   console.log('First Listener Executed');
}

function listenerTwo() {
   console.log('Second Listener Executed');
}

eventEmitter.on('listenerOne', listenerOne); // Register for listenerOne
eventEmitter.on('listenerOne', listenerTwo); // Register for listenerOne

// When the event "listenerOne" is emitted, both the above callbacks should be invoked.
eventEmitter.emit('listenerOne');

/* Output
First Listener Executed
Second Listener Executed */

/**
 * Emit Events Once
 */
const events = require('events');
// const eventEmitter = new events.EventEmitter();

function listenerOnce() {
   console.log('listenerOnce fired once');
}

eventEmitter.once('listenerOne', listenerOnce); // Register listenerOnce
eventEmitter.emit('listenerOne');

// Output
// listenerOnce fired once

/**
 * Event loop in Node.js
 */
const events = require('events');
const eventEmitter = new events.EventEmitter();

// Create an event handler as follows
const connectHandler = function connected() {
   console.log('connection succesful.');
   eventEmitter.emit('data_received');
}

// Bind the connection event with the handler
eventEmitter.on('connection', connectHandler);
 
// Bind the data_received event with the anonymous function
eventEmitter.on('data_received', function() {
   console.log('data received succesfully.');
});

// Fire the connection event 
eventEmitter.emit('connection');
console.log("Program Ended.");

/* Output
Connection succesful.
Data received succesfully.
Program Ended.
*/

/**
 * setImmediate() and process.nextTick()
 */
setImmediate(() => {
  console.log("1st Immediate");
});

setImmediate(() => {
  console.log("2nd Immediate");
});

process.nextTick(() => {
  console.log("1st Process");
});

process.nextTick(() => {
  console.log("2nd Process");
});

// First event queue ends here
console.log("Program Started");

/* Output
Program Started
1st Process
2nd Process
1st Immediate
2nd Immediate
*/

/**
 * Callback Function
 */
function message(name, callback) {
  console.log("Hi" + " " + name);
  callback();
}

// Callback function
function callMe() {
  console.log("I am callback function");
}

// Passing function as an argument
message("Node.JS", callMe);

/* Output
Hi Node.JS
I am callback function
*/

/**
 * Events Module
 */
const event = require('events');  
// const eventEmitter = new event.EventEmitter();  
  
// add listener function for Sum event  
eventEmitter.on('Sum', function(num1, num2) {  
    console.log('Total: ' + (num1 + num2));  
});  

// call event  
eventEmitter.emit('Sum', 10, 20);

// Output
// Total: 30

/**
 * Callbacks
 */
function sum(number) {
  console.log('Total: ' + number);
}

function calculator(num1, num2, callback) {
  let total = num1 + num2;
  callback(total);
}

calculator(10, 20, sum);

// Output
// Total: 30

/**
 * Error First Callback
 */

fs.readFile( "file.json", function ( err, data ) {
  if ( err ) {
    console.error( err );
  }
  console.log( data );
});

/**
 * Promises
 */
const myPromise = new Promise((resolve, reject) => {
  setTimeout(() => {
    resolve("Successful!");
  }, 300);
});

/**
 * Async Await
 */
const getrandomnumber = function(){
  return new Promise((resolve, reject)=>{
      setTimeout(() => {
          resolve(Math.floor(Math.random() * 20));
      }, 1000);
  });
}

const addRandomNumber = async function(){
  const sum = await getrandomnumber() + await getrandomnumber();
  console.log(sum);
}

addRandomNumber();


/**
 * Callback Handler
 */
const Division = (numerator, denominator, callback) => {
  if (denominator === 0) {
    callback(new Error('Divide by zero error!'));
  } else {
    callback(null, numerator / denominator);
  }
};

// Function Call
Division(5, 0, (err, result) => {
if (err) {
  return console.log(err.message);
}
console.log(`Result: ${result}`);
});

/**
 * Timing
 */

// Setting timeout for the function
setTimeout(function () {
  console.log('setTimeout() function running...');
}, 500);

// Running this function immediately before any other
setImmediate(function () {
 console.log('setImmediate() function running...');
});

// Directly printing the statement
console.log('Normal statement in the event loop');

// Output
// Normal statement in the event loop
// setImmediate() function running...
// setTimeout() function running...

setInterval(function() {
  console.log('Display this Message intervals of 1 seconds!');
}, 1000);

/**
 * Implement Sleep
 */

function delay(time) {
  return new Promise((resolve) => setTimeout(resolve, time));
}

async function run() {
  await delay(1000);
  console.log("This printed after about 1 second");
}

run();

/**
 * read_file.js
 */
const http = require('http');
const fs = require('fs');

http.createServer(function (req, res) {
  fs.readFile('index.html', function(err, data) {
    res.writeHead(200, {'Content-Type': 'text/html'});
    res.write(data);
    res.end();
  });
}).listen(3000);

/**
 * NODE.JS STREAMS
 */
// Read
const fs = require("fs");
let data = "";

// Create a readable stream
const readerStream = fs.createReadStream("file.txt");

// Set the encoding to be utf8.
readerStream.setEncoding("UTF8");

// Handle stream events --> data, end, and error
readerStream.on("data", function (chunk) {
  data += chunk;
});

readerStream.on("end", function () {
  console.log(data);
});

readerStream.on("error", function (err) {
  console.log(err.stack);
});

// Write
const fs = require("fs");
//const data = "File writing to a stream example";

// Create a writable stream
const writerStream = fs.createWriteStream("file.txt");

// Write the data to stream with encoding to be utf8
writerStream.write(data, "UTF8");

// Mark the end of file
writerStream.end();

// Handle stream events --> finish, and error
writerStream.on("finish", function () {
  console.log("Write completed.");
});

writerStream.on("error", function (err) {
  console.log(err.stack);
});

// Piping
const fs = require("fs");

// Create a readable stream
// const readerStream = fs.createReadStream('input.txt');

// Create a writable stream
// const writerStream = fs.createWriteStream('output.txt');

// Pipe the read and write operations
// read input.txt and write data to output.txt
readerStream.pipe(writerStream);

// Chaining
const fs = require("fs");
const zlib = require('zlib');

// Compress the file input.txt to input.txt.gz
fs.createReadStream('input.txt')
   .pipe(zlib.createGzip())
   .pipe(fs.createWriteStream('input.txt.gz'));
  
console.log("File Compressed.");

/**
 * Cluster Module
 */
const cluster = require("cluster");

if (cluster.isMaster) {
  console.log(`Master process is running...`);
  cluster.fork();
  cluster.fork();
} else {
  console.log(`Worker process started running`);
}

/* Output
Master process is running...
Worker process started running
Worker process started running
*/

/**
 * Server Load Balancing in Node.js
 */
const cluster = require("cluster");
const express = require("express");
const os = require("os");

if (cluster.isMaster) {
  console.log(`Master PID ${process.pid} is running`);

  // Get the number of available cpu cores
  const nCPUs = os.cpus().length;
  // Fork worker processes for each available CPU core
  for (let i = 0; i < nCPUs; i++) {
    cluster.fork();
  }

  cluster.on("exit", (worker, code, signal) => {
    console.log(`Worker PID ${worker.process.pid} died`);
  });
} else {
  // Workers can share any TCP connection
  // In this case it is an Express server
  const app = express();
  app.get("/", (req, res) => {
    res.send("Node is Running...");
  });

  app.listen(3000, () => {
    console.log(`App listening at http://localhost:3000/`);
  });

  console.log(`Worker PID ${process.pid} started`);
}

/* Output
Master PID 13972 is running
Worker PID 5680 started
App listening at http://localhost:3000/
Worker PID 14796 started
...
*/

/* 2 Express servers load balancer */
const body = require('body-parser');
const express = require('express');

const app1 = express();
const app2 = express();

// Parse the request body as JSON
app1.use(body.json());
app2.use(body.json());

const handler = serverNum => (req, res) => {
  console.log(`server ${serverNum}`, req.method, req.url, req.body);
  res.send(`Hello from server ${serverNum}!`);
};

// Only handle GET and POST requests
app1.get('*', handler(1)).post('*', handler(1));
app2.get('*', handler(2)).post('*', handler(2));

app1.listen(3000);
app2.listen(3001);

/* JSON Web Token (JWT) for authentication in Node.js */
// npm install jsonwebtoken bcryptjs --save

/**
 * AuthController.js
 */
const express = require('express');
const router = express.Router();
const bodyParser = require('body-parser');
const User = require('../user/User');

const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const config = require('../config');


router.use(bodyParser.urlencoded({ extended: false }));
router.use(bodyParser.json());

router.post('/register', function(req, res) {
  
  let hashedPassword = bcrypt.hashSync(req.body.password, 8);
  
  User.create({
    name : req.body.name,
    email : req.body.email,
    password : hashedPassword
  },
  function (err, user) {
    if (err) return res.status(500).send("There was a problem registering the user.")
    // create a token
    let token = jwt.sign({ id: user._id }, config.secret, {
      expiresIn: 86400 // expires in 24 hours
    });
    res.status(200).send({ auth: true, token: token });
  });
});

/**
 * config.js
 */
module.exports = {
  'secret': 'supersecret'
};

/* microservices architecture */
//  server.js

const express = require('express')
// const app = express();
const port = process.env.PORT || 3000;

const routes = require('./api/routes');
routes(app);
app.listen(port, function() {
   console.log('Server started on port: ' + port);
});

// Routes
const controller = require('./controller');

module.exports = function(app) {
   app.route('/about')
       .get(controller.about);
   app.route('/distance/:zipcode1/:zipcode2')
       .get(controller.getDistance);
};

// Controllers
const properties = require('../package.json')
const distance = require('../service/distance');

const controllers = {
   about: function(req, res) {
       let aboutInfo = {
           name: properties.name,
           version: properties.version
       }
       res.json(aboutInfo);
   },
   getDistance: function(req, res) {
           distance.find(req, res, function(err, dist) {
               if (err)
                   res.send(err);
               res.json(dist);
           });
       },
};

module.exports = controllers;

/**
 * myLogger middleware
 */
const express = require("express");
// const app = express();

const myLogger = function (req, res, next) {
  console.log("LOGGED");
  next();
};

app.use(myLogger);

app.get("/", (req, res) => {
  res.send("Hello World!");
});

app.listen(3000);

/**
 * Simple server using Express.js
 */
const express = require("express");
// const app = express();

app.get("/", function (req, res) {
  res.send("Hello World!");
});

const server = app.listen(3000, function () {});

/* separate Express 'app' and 'server' */
/**
 * app.js
 */
//const app = express();

app.use(bodyParser.json());
app.use("/api/events", events.API);
app.use("/api/forms", forms);

/**
 * server.js
 */
// const app = require('../app');
const http = require('http');


// Get port from environment and store in Express.
// const port = normalizePort(process.env.PORT || '3000');
app.set('port', port);


// Create HTTP server.
// const server = http.createServer(app);

/**
 * Joi - Validating input
 */
const Joi = require('joi');

const schema = Joi.object().keys({
    username: Joi.string().alphanum().min(3).max(30).required(),
    password: Joi.string().regex(/^[a-zA-Z0-9]{3,30}$/),
    access_token: [Joi.string(), Joi.number()],
    birthyear: Joi.number().integer().min(1900).max(2013),
    email: Joi.string().email()
}).with('username', 'birthyear').without('password', 'access_token')

// Return result
const result = Joi.validate({
    username: 'abc',
    birthyear: 1994
}, schema)
// result.error === null -> valid

/* Security.txt */
const express = require('express')
const securityTxt = require('express-security.txt')

// const app = express()

app.get('/security.txt', securityTxt({
  // your security address
  contact: 'email@example.com',
  // your pgp key
  encryption: 'encryption',
  // if you have a hall of fame for securty resourcers, include the link here
  acknowledgements: 'http://acknowledgements.example.com'
}))

/* handle file upload in Node.js */
// npm install express body-parser multer --save
/**
 * File Upload in Node.js
 */
/* server.js */
const express = require("express");
const bodyParser = require("body-parser");
const multer = require("multer");
// const app = express();

// for text/number data transfer between clientg and server
app.use(bodyParser());

const storage = multer.diskStorage({
  destination: function (req, file, callback) {
    callback(null, "./uploads");
  },
  filename: function (req, file, callback) {
    callback(null, file.fieldname + "-" + Date.now());
  },
});

const upload = multer({ storage: storage }).single("userPhoto");

app.get("/", function (req, res) {
  res.sendFile(__dirname + "/index.html");
});

// POST: upload for single file upload
app.post("/api/photo", function (req, res) {
  upload(req, res, function (err) {
    if (err) {
      return res.end("Error uploading file.");
    }
    res.end("File is uploaded");
  });
});

app.listen(3000, function () {
  console.log("Listening on port 3000");
});

/* index.html 
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Multer-File-Upload</title>
</head>
<body>
    <h1>MULTER File Upload | Single File Upload</h1> 

    <form id = "uploadForm"
         enctype = "multipart/form-data"
         action = "/api/photo"
         method = "post"
    >
      <input type="file" name="userPhoto" />
      <input type="submit" value="Upload Image" name="submit">
    </form>
</body>
</html>
*/

/**
 * body-parser
 */
//npm install body-parser
const express = require("express");
const bodyParser = require("body-parser");

// const app = express();

// create application/json parser
const jsonParser = bodyParser.json();

// create application/x-www-form-urlencoded parser
const urlencodedParser = bodyParser.urlencoded({ extended: false });

// POST /login gets urlencoded bodies
app.post("/login", urlencodedParser, function (req, res) {
  res.send("welcome, " + req.body.username);
});

// POST /api/users gets JSON bodies
app.post("/api/users", jsonParser, function (req, res) {
  // create user in req.body
});

/**
 * cookie-parser
 */
// npm install cookie-parser
const express = require('express')
const cookieParser = require('cookie-parser')

// const app = express()
app.use(cookieParser())

app.get('/', function (req, res) {
  // Cookies that have not been signed
  console.log('Cookies: ', req.cookies)

  // Cookies that have been signed
  console.log('Signed Cookies: ', req.signedCookies)
})

app.listen(3000)

/* morgan - HTTP request logger middleware for node.js. */
// npm install morgan
/**
 * Writing logs to a file
 */
const express = require('express')
const fs = require('fs')
const morgan = require('morgan')
const path = require('path')

// const app = express()

// create a write stream (in append mode)
const accessLogStream = fs.createWriteStream(path.join(__dirname, 'access.log'), { flags: 'a' })

// setup the logger
app.use(morgan('combined', { stream: accessLogStream }))

app.get('/', function (req, res) {
  res.send('hello, world!')
})

/* cors */
// npm install cors
/**
 * Enable CORS for a Single Route
 */
const express = require('express')
const cors = require('cors')
// const app = express()

app.get('/products/:id', cors(), function (req, res, next) {
  res.json({msg: 'This is CORS-enabled for a Single Route'})
})

app.listen(8080, function () {
  console.log('CORS-enabled web server listening on port 80')
})

/* dotenv */
// npm install dotenv
// .env

DB_HOST=localhost
DB_USER=admin
DB_PASS=root

/**
 * config.js
 */
const db = require('db')

db.connect({
  host: process.env.DB_HOST,
  username: process.env.DB_USER,
  password: process.env.DB_PASS
})

/* A JavaScript date library for parsing, validating, manipulating, and formatting dates. */
// npm install moment --save
const moment = require('moment');

moment().format('MMMM Do YYYY, h:mm:ss a'); // October 24th 2022, 3:15:22 pm
moment().format('dddd');                    // Saturday
moment().format("MMM Do YY");               // Oct 24th 22

moment("20111031", "YYYYMMDD").fromNow(); // 9 years ago
moment("20120620", "YYYYMMDD").fromNow(); // 8 years ago
moment().startOf('day').fromNow();        // 15 hours ago

moment().subtract(10, 'days').calendar(); // 10/14/2022
moment().subtract(6, 'days').calendar();  // Last Sunday at 3:18 PM
moment().subtract(3, 'days').calendar();  // Last Wednesday at 3:18 PM

/* RESTful Web Services */
// users.json
const users = {
  "user1" : {
     "id": 1,
     "name" : "Ehsan Philip",
     "age" : 24
  },

  "user2" : {
     "id": 2,
     "name" : "Karim Jimenez",
     "age" : 22
  },

  "user3" : {
     "id": 3,
     "name" : "Giacomo Weir",
     "age" : 18
  }
}

// GET user
const express = require('express');
// const app = express();
const fs = require("fs");

app.get('/listUsers', function (req, res) {
   fs.readFile( __dirname + "/" + "users.json", 'utf8', function (err, data) {
      console.log( data );
      res.end( data );
   });
})


// POST add user
const user = {
  "user4" : {
     "id": 4,
     "name" : "Spencer Amos",
     "age" : 28
  }
}

app.post('/addUser', function (req, res) {
  // First read existing users.
  fs.readFile( __dirname + "/" + "users.json", 'utf8', function (err, data) {
     data = JSON.parse( data );
     data["user4"] = user["user4"];
     console.log( data );
     res.end( JSON.stringify(data));
  });
})

// delete user
const id = 2;

app.delete('/deleteUser', function (req, res) {
   // First read existing users.
   fs.readFile( __dirname + "/" + "users.json", 'utf8', function (err, data) {
      data = JSON.parse( data );
      delete data["user" + 2];
      console.log( data );
      res.end( JSON.stringify(data));
   });
})

/*
const server = app.listen(3000, function () {
   const host = server.address().address
   const port = server.address().port
   console.log("App listening at http://%s:%s", host, port)
});
*/

/**
 * req.params
 */

// GET  http://localhost:3000/employees/10

app.get('/employees/:id', (req, res, next) => {
  console.log(req.params.id); // 10
})

/**
 * req.query
 */

// GET  http://localhost:3000/employees?page=20

app.get('/employees', (req, res, next) => {
  console.log(req.query.page) // 20
})

/**
 * Promise
 */
function getSum(num1, num2) {
  const myPromise = new Promise((resolve, reject) => {
    if (!isNaN(num1) && !isNaN(num2)) {
      resolve(num1 + num2);
    } else {
      reject(new Error("Not a valid number"));
    }
  });

  return myPromise;
}

console.log(getSum(10, 20)); // Promise { 30 }

/**
 * POST Request using Axios
 */
const express = require("express");
// const app = express();
const axios = require("axios");

app.post("/user", async (req, res) => {
  try {
    const payload = { name: "Aashita Iyer", email: "aashita.iyer@email.com" };
    const response = await axios.post("http://httpbin.org/post", payload);
    console.log(response.data);
    res.status(200).json(response.data);
  } catch (err) {
    res.status(500).json({ message: err });
  }
});

app.listen(3000, function () {
  console.log(`App listening at http://localhost:3000/`);
});

// Blocking
const fs = require('fs');
//const data = fs.readFileSync('/file.md'); // blocks here until file is read
console.log(data);
moreWork(); // will run after console.log

// Non-blocking
const fs = require('fs');
fs.readFile('/file.md', (err, data) => {
  if (err) throw err;
  console.log(data);
});
moreWork(); // will run before console.log

/* HTTP request */
const request = require('request');

const options = {
    url: 'https://nodejs.org/file.json',
    method: 'GET',
    headers: {
        'Accept': 'application/json',
        'Accept-Charset': 'utf-8',
        'User-Agent': 'my-reddit-client'
    }
};

request(options, function(err, res, body) {
    let json = JSON.parse(body);
    console.log(json);
});

/* promise vs async - await */
function logFetch(url) {
  return fetch(url)
    .then(response => {
      console.log(response);
    })
    .catch(err => {
      console.error('fetch failed', err);
    });
}

async function logFetch(url) {
  try {
    const response = await fetch(url);
    console.log(response);
  }
  catch (err) {
    console.log('fetch failed', err);
  }
}

function doubleAfter2Seconds(x) {
  return new Promise(resolve => {
    setTimeout(() => {
      resolve(x * 2);
    }, 2000);
  });
}

async function addAsync(x) {
  const a = await doubleAfter2Seconds(10);
  const b = await doubleAfter2Seconds(20);
  const c = await doubleAfter2Seconds(30);
  return x + a + b + c;
}


addAsync(10).then((sum) => {
  console.log(sum);
});

/**
 * Get Request using Axios
 */
const express = require("express");
// const app = express();
const axios = require("axios");

app.get("/async", async (req, res) => {
  try {
    const response = await axios.get("https://jsonplaceholder.typicode.com/todos/1");
    res.status(200).json(response.data);
  } catch (err) {
    res.status(500).json({ message: err });
  }
});

app.listen(3000, function () {
  console.log(`App listening at http://localhost:3000/`);
});

// GET method route
app.get('/', function (req, res) {
  res.send('GET request')
})

// POST method route
app.post('/login', function (req, res) {
  res.send('POST request')
})

// ALL method route
app.all('/secret', function (req, res, next) {
  console.log('Accessing the secret section ...')
  next() // pass control to the next handler
})

// This route path will match requests to /about.
app.get('/about', function (req, res) {
  res.send('about')
})


// This route path will match acd and abcd.
app.get('/ab?cd', function (req, res) {
  res.send('ab?cd')
})


// This route path will match butterfly and dragonfly
app.get(/.*fly$/, function (req, res) {
  res.send('/.*fly$/')
})

// route params
app.get('/users/:userId', function (req, res) {
  res.send(req.params)
})

const express = require('express')
// const router = express.Router()

// middleware that is specific to this router
router.use(function timeLog (req, res, next) {
  console.log('Time: ', Date.now())
  next()
})

// define the home page route
router.get('/', function (req, res) {
  res.send('Birds home page')
})

// define the about route
router.get('/about', function (req, res) {
  res.send('About birds')
})

module.exports = router

/* cache data in Node.js */
// npm install express node-cache axios

/**
 * In-Memory Cache 
 */
const express = require("express");
const NodeCache = require("node-cache");
const axios = require("axios");

// const app = express();
// const cache = new NodeCache({ stdTTL: 15 });

/**
 * GET Cached Data
 */
const verifyCache = (req, res, next) => {
  try {
    const { id } = req.params;
    if (cache.has(id)) {
      return res.status(200).json(cache.get(id));
    }
    return next();
  } catch (err) {
    throw new Error(err);
  }
};

app.get("/", (req, res) => {
  return res.json({ message: "Hello World" });
});

/**
 * GET ToDo Items
 */
app.get("/todos/:id", verifyCache, async (req, res) => {
  try {
    const { id } = req.params;
    const { data } = await axios.get(`https://jsonplaceholder.typicode.com/todos/${id}`);
    cache.set(id, data);
    return res.status(200).json(data);
  } catch ({ response }) {
    return res.sendStatus(response.status);
  }
});

app.listen(3000, function () {
  console.log(`App listening at http://localhost:3000/`);
});

/* implement caching using Redis */
// npm install -save redis

const express = require("express");
const axios = require("axios");
const redis = require("redis");
const app = express();

const client = redis.createClient(6379);

client.on("error", (error) => {
  console.error(error);
});

app.get("/", (req, res) => {
  return res.json({ message: "Hello World" });
});

const cache = (req, res, next) => {
  try {
    const { id } = req.params;
    client.get(id, (error, result) => {
      if (error) throw error;
      if (result !== null) {
        return res.json(JSON.parse(result));
      } else {
        return next();
      }
    });
  } catch (err) {
    throw new Error(err);
  }
};

app.get("/todos/:id", cache, async (req, res) => {
  try {
    const { id } = req.params;
    const data = await axios.get(`https://jsonplaceholder.typicode.com/todos/${id}`);
    client.set(id, JSON.stringify(data), "ex", 15);
    return res.status(200).json(data);
  } catch ({ response }) {
    return res.sendStatus(response.status);
  }
});

app.listen(3000, function () {
  console.log(`App listening at http://localhost:3000/`);
});

/* implement Memcached */
// npm install memcached

/**
 * Memcached
 */
const Memcached = require('memcached');
// all global configurations should be applied to the .config object of the Client.
Memcached.config.poolSize = 25;

const memcached = new Memcached('localhost:11211', { retries:10, retry:10000, remove:true, failOverServers:['192.168.0.103:11211']});

/* Error Handling approaches */
// Using try-catch block
function square(num) {
  if (typeof num !== "number") {
    throw new TypeError(`Expected number but got: ${typeof num}`);
  }

  return num * num;
}

try {
  square("10");
} catch (err) {
  console.log(err.message); // Expected number but got: string
}

// Using promises
function square(num) {
  return new Promise((resolve, reject) => {
    setTimeout(() => {
      if (typeof num !== "number") {
        reject(new TypeError(`Expected number but got: ${typeof num}`));
      }

      const result = num * num;
      resolve(result);
    }, 100);
  });
}

square("10")
  .then((result) => console.log(result))
  .catch((err) => console.error(err));

// Error-first callbacks
const fs = require('fs');

fs.readFile('/path/to/file.txt', (err, result) => {
  if (err) {
    console.error(err);
    return;
  }

  // Log the file contents if no error
  console.log(result);
});

// Using the async/await
const fs = require('fs');
const util = require('util');

const readFile = util.promisify(fs.readFile);

const read = async () => {
  try {
    const result = await readFile('/path/to/file.txt');
    console.log(result);
  } catch (err) {
    console.error(err);
  }
};

read();

// middlewares
const winston = require("winston");

const logger = winston.createLogger({
  level: "debug",
  format: winston.format.json(),
  transports: [new winston.transports.Console()],
});

module.exports = logger;

// ---
const express = require("express");
const logger = require("./logger");
// const app = express();

app.get("/event", (req, res, next) => {
  try {
    throw new Error("Not User!");
  } catch (error) {
    logger.error("Events Error: Unauthenticated user");
    res.status(500).send("Error!");
  }
});

app.listen(3000, () => {
  logger.info("Server Listening On Port 3000");
});

/* Joi module for schema validation */
const Joi = require("joi");

// User-defined function to validate the user

function validateUser(user) {

  const JoiSchema = Joi.object({

    username: Joi.string().min(5).max(30).required(),

    email: Joi.string().email().min(5).max(50).optional(),

    date_of_birth: Joi.date().optional(),

    account_status: Joi.string()
      .valid("activated")
      .valid("unactivated")
      .optional(),
  }).options({ abortEarly: false });

  return JoiSchema.validate(user);
}

/* 
const user = {
  username: "Deepak Lucky",
  email: "deepak.lucky@gmail.com",
  date_of_birth: "2000-07-07",
  account_status: "activated",
};
*/

let response = validateUser(user);

if (response.error) {
  console.log(response.error.details);
} else {
  console.log("Validated Data");
}

/* crypto in Node.js */
// encryption using hash
const crypto = require('crypto');  
const secret = 'abcdefg';  
const hash = crypto.createHmac('sha256', secret)  
                   .update('Welcome to Node.js')  
                   .digest('hex');  
console.log(hash);  

// encrypt using Cipher
const crypto = require('crypto');  
const cipher = crypto.createCipher('aes192', 'a password');  

const encrypted = cipher.update('Hello Node.js', 'utf8', 'hex');  
encrypted += cipher.final('hex');  

console.log(encrypted);

// decrypt Cipher
const crypto = require('crypto');  
const decipher = crypto.createDecipher('aes192', 'a password');  

//const encrypted = '4ce3b761d58398aed30d5af898a0656a3174d9c7d7502e781e83cf6b9fb836d5';  
const decrypted = decipher.update(encrypted, 'hex', 'utf8');  
decrypted += decipher.final('utf8');  

console.log(decrypted);  

/* gracefully shutdown Node.js Server */
function shutdown() {
  server.close(function onServerClosed(err) {
    if (err) {
      console.error(err);
      process.exit(1);
    }

    closeMyResources(function onResourcesClosed(err) {
      // error handling
      process.exit();
    });
  });
}

process.on("SIGTERM", function onSigterm() {
  console.info("Got SIGTERM. Graceful shutdown start",  new Date().toISOString());
  // start graceul shutdown here
  shutdown();
});