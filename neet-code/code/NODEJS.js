// Load the file system module
const fs = require("fs");

// While waiting for the file to be read, can't do anything else!
const data = fs.readFileSync("/file.md", "utf-8");
console.log(data);

// Will only run after the console.log on the previous line
doMoreStuff();

// Load the file system module
const fs = require("fs");

// Pass a callback function as a third argument
fs.readFile("/file.md", "utf-8", (err, data) => {
  if (err) throw err;
  console.log(data);
});

// Will run before console.log
doMoreStuff();

//FS Module
const fs = require("fs").promises;
const getStats = async (path) => {
  try {
    // Pass in the file path
    const stats = await fs.stat(path);
    console.log(stats);
  } catch (error) {
    console.error(error);
  }
};

getStats("./test.txt");

const readFile = async (path) => {
  try {
    const contents = await fs.readFile(path, "utf8");
    console.log(contents);
  } catch (error) {
    console.error(error);
  }
};

readFile("./test.txt");

//
const writeFile = async (path, data) => {
  try {
    await fs.writeFile(path, data);
  } catch (error) {
    console.error(error);
  }
};

const appendFile = async (path, data) => {
  try {
    await fs.appendFile(path, data);
  } catch (error) {
    console.error(error);
  }
};
appendFile("./test.txt", "appending another hello world");

//Process module
const name1 = process.argv[2];
console.log(process.argv);
const location = process.argv[3];
console.log(`Hi, I'm ${name1}! I live in ${location}`);
//node process.js Hou Brooklyn

//Node.js environment variables
console.log("process.env:", process.env);
console.log("process.env.NODE_ENV: ", process.env.NODE_ENV);

//Template Literals
const greeting = "Good morning";
const weather = "sunny";

// ES5
const sentence =
  greeting + ", how are you doing?" + " The weather is " + weather + " today.";

// ES6
const sentenceES6 = `${greeting}, how are you doing? The weather is ${weather} today.`;
//Arrow function
const printName1 = (firstName, lastName) => {
  return `${firstName}${lastName}`;
};

const printName2 = (firstName, lastName) => `${firstName}${lastName}`;

//Default
function greeting(name = "Jane", greeting) {
  return `${greeting}, ${name}! `;
}

//***Destructuring
const person = {
  name: "Hou Chia",
  title: "software engineer",
  city: "Brooklyn,NY",
  age: 32,
};
const { name, title, city, age } = person;

const { //Different name
  name: employeeName,
  title: employeeTitle,
  city: employeeCity,
  age: employeeAge,
} = person;

//Array
const letters = ["a", "b", "c"];

const [firstLetter, secondLetter, thirdLetter] = letters;
let [a = 4, b = 3] = [2]; //Default value

//Destructuring parameters
const person = {
  name: "Hou Chia",
  title: "software engineer",
  city: "Brooklyn,NY",
  age: 32,
};

const introduce = ({ name, title, city, age }) => {
  return `Hello, my name is ${name}, and I'm a ${title}. I live in ${city}, and I'm ${age} years old.`;
};

/* Rest spread */
//Rest
let { a, b, ...pairs } = { a: 10, b: 20, c: 30, d: 40 };
console.log(a); // 10
console.log(b); // 20
console.log(pairs); // { c: 30, d: 40 }
//Spread 
const person = { name: "Hou", title: "software engineer" };
const personalInfo = { age: 32, location: "Brooklyn, NY" };

const employee = {
  id: 1,
  department: "engineering",
  ...person,
  ...personalInfo,
};

/* Module */
// CommonJS
const { sayHi, person, greetingInMandarin } = require("./file1.js");
console.log(sayHi(greetingInMandarin, person.firstName, person.lastName));

onst sayHi = (greeting, firstName, lastName) => {
  return `${greeting}, ${firstName}, ${lastName}!`;
};

const person = {
  firstName: "Hou",
  lastName: "Chia",
};

let greetingInMandarin = "Ni hao";

// can export function, const, let
module.exports = { sayHi, person, greetingInMandarin  };
// default
export default function greet() {
  return "hello";
}

/* Enhanced Object Literal */
//Shorthand for initializing property values
// ES5
function getCar(make, model, year) {
  return {
    make: make,
    model: model,
    year: year,
  };
}

// ES6
function getCar(make, model, year) {
  return {
    make, // Notice you can drop the `: make` part, since the property name matches the variable name
    model,
    year,
  };
}
//Shorthand method
// ES5
var server = {
  name: "Server",
  restart: function () {
    console.log("The" + this.name + " is restarting...");
  },
  stop: function () {
    console.log("The" + this.name + " is stopping...");
  },
};

// ES6
const server = {
  name: "Server",
  restart() {
    console.log(`The ${this.name} is restarting...`);
  },
  stop() {
    console.log(`The ${this.name} is stopping...`);
  },
};

//Computed property key
const fieldName = "location";

const formData = {
  firstName: "Hou",
  lastName: "Chia",
  [fieldName]: "Cleveland",
};

console.log(formData.location); // prints "Cleveland"

/* Async */
const greet = async () => "hello"; // async implicitly wraps 'hello' in a promise
console.log(greet()); // returns a Promise object

greet().then((greeting) => console.log(greeting)); // logs 'hello'
//~ old
const greet = async () => Promise.resolve("hello").then(function(value) {
  console.log(value); // "Success"
}, function(value) {
  // not called
});
console.log(greet()); // returns a Promise object

// await
const fetchTrivia = async () => {
  try {
    const response = await fetch(
      "https://opentdb.com/api.php?amount=1&category=18"
    );
    const data = await response.json();
    console.log(data.results[0]);
  } catch (error) {
    console.error(error);
  }
};

fetchTrivia();

//old
const fetchTrivia = () => {
  fetch("https://opentdb.com/api.php?amount=1&category=18")
    .then((response) => response.json())
    .then((data) => {
      console.log(data.results[0]);
    })
    .catch(console.error);
};

fetchTrivia();

/* Closure */
const lastName = "Chia";

const printPerson = () => {
  const firstName = "Hou";

  const logNameAndLocation = () => {
    const location = "Brooklyn, NY";
    console.log(`${firstName} ${lastName}, ${location}`);
  };
  return logNameAndLocation;
};

const printPersonNameAndLocation = printPerson();
printPersonNameAndLocation();

