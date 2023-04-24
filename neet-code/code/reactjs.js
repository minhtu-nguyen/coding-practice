// *** Fundamentals
//Operator
//Arithmetic Operators
console.log("****Arithmetic Operators****\n")
console.log("2 + 3 = " + (2 + 3))
console.log("2 - 3 = " + (2 - 3))
console.log("2 * 3 = " + (2 * 3))
console.log("6 / 3 = " + (6 / 3))
console.log("7 / 3 = " + (7 / 3))
//Assignment Operators
console.log("\n****Assignment Operators****\n")
var x = 3;
console.log("x = " + x)
console.log("x += 1 gives x = " + (x+=1)) // adds 1 = value of x=4
console.log("x -= 1 gives x = " + (x-=1)) // subtracts 1 = value of x=3
console.log("x *= 3 gives x = " + (x*=3)) // multiplies 3 with x = value of x=9
console.log("x /= 3 gives x = " + (x/=3)) // Divides 3 with x = value of x=3
//Logical Operators
console.log("\n****Logical Operators****\n")
console.log("1 OR 1 = " + (1 || 1)) // 1 OR 1
console.log("1 OR 0 = " + (1 || 0)) // 1 OR 0
console.log("0 OR 0 = " + (0 || 0)) // 0 OR 0
console.log("1 AND 1 = " + (1 && 1)) // 1 AND 1
console.log("1 AND 0 = " + (1 && 0)) // 1 AND 0
console.log("0 AND 0 = " + (0 && 0)) // 0 AND 0
console.log(!true)  // NOT TRUE
console.log(!1)     // NOT TRUE
console.log(!false) // NOT FALSE
console.log(!0)     // NOT FALSE
//Comma operator
console.log("\n****Comma Operator****")
var a = 4;
a = (a++, a);
console.log("The value for expression with comma operator is: " + a) //returns 5
//Comparison operators
console.log("\n****Comparison Operators****")
console.log(1 > 2) //false
console.log(1 < 2) //true
console.log(1 == 1) //true
console.log(1 != 1) //false
//Bitwise Operator
console.log("\n****Bitwise Operators****")
console.log("Bitwise AND of 5 and 1: " + (5 & 1)) //returns 1
console.log("Bitwise OR of 5 and 1: " + (5 | 1)) // returns 5 
console.log("Bitwise XOR of 5 and 1: " + (5 ^ 1)) //returns 4
//String Operator
console.log("\n****String Operator****")
console.log("Concatenation" + " (+)" + " operator in action")
//Conditional Operator
console.log("\n ****Conditional Operator****")
var num_of_months = 13
var ans = (num_of_months > 12) ? "Invalid" : "Valid"
console.log(ans) //Returns Invalid

// Ternary Operator
import React from 'react';

export default class App extends React.Component {
  render() {
    const users = [
      { name: 'Robin' },
      { name: 'Markus' },
    ];

    const showUsers = true;
    if (!showUsers) {
      return null;
    }

    return (
      <ul>
        {users.map(user => <li>{user.name}</li>)}
      </ul>
    );
  }
}

export default class App extends React.Component {
  render() {
       const users = [
      { name: 'Robin' },
      { name: 'Markus' },
    ];
    const showUsers = true;
    return (
      <div>
        {
          !showUsers ? (
            <ul>
              {users.map(user => <li>{user.name}</li>)}
            </ul>
          ) : (
            null
          )
        }
      </div>
    );
  }
}

// Destructuring
// no destructuring
const users = this.state.users;
const counter = this.state.counter;

// destructuring
const { users, counter } = this.state;

// no destructuring
function Greeting(props) {
  return <h1>{props.greeting}</h1>;
}

// destructuring
function Greeting({ greeting }) {
  return <h1>{greeting}</h1>;
}
// rest destructuring
const { users, ...rest } = this.state;

// Spread Operator
a = [1,2,3];
b = [4,5,6];
c = [...a, ...b]; //spread operator
console.log("c: " + c);

// anonymous function
var myFunction = function()
{
  console.log("Hello! I'm an Anonymous function");
}

//Arrow Functions
// JavaScript ES6 arrow function with body
const getGreetingArrow1 = () => {
  return 'Welcome to JavaScript';
}

// JavaScript ES6 arrow function without body and implicit return
const getGreetingArrow2 = () =>
  'Welcome to JavaScript';

const students = [
  { ID: 1, present: true},
  { ID: 2, present: true},
  { ID: 3, present: false}, 
];

const presentStudents = students.filter(student => student.present);
console.log(presentStudents);

// Higher Order Functions
import React from 'react';

const doFilter = query => user =>
   query === user.name;

export default class App extends React.Component {

  constructor(props){
    super(props);  
    
    this.state = {
    query: '',
    };
    
    this.onChange=this.onChange.bind(this);
  }
  
  onChange(event) {
    this.setState({ query: event.target.value });
  }
  
  render() {
  const users = [
      { name: 'Robin' },
      { name: 'Markus' },
    
    ];
    return (
      <div>
        <ul>
          { users
            .filter(doFilter(this.state.query))
            .map(myuser => <li>{myuser.name}</li>)
          }
        </ul>
        <input
          type="text"
          onChange={this.onChange}
        />
      </div>
    );
  }
}

//Map, Filter
import React from 'react';

export default class App extends React.Component {
  render() {
    var users = [
      { name: 'Robin', isDeveloper: true },
      { name: 'Markus', isDeveloper: false },
      { name: 'John', isDeveloper: true },
    ];

    return (
      <ul>
        {users
          .filter(user => user.isDeveloper)
          .map(user => <li>{user.name}</li>)
        }
      </ul>
    );
  }
}

// Class
class Developer {
  constructor(firstname, lastname) {
    this.firstname = firstname;
    this.lastname = lastname;
  }

  getName() {
    return `${this.firstname} ${this.lastname}`;
  }
}

var me = new Developer('Robin', 'Wieruch');

console.log(me.getName());

// Object
let computer = { brand : 'HP', RAM : '8 GB', clockspeed : "2 GHz"};

// object definitions can have spaces and newlines!
let computer2 = { 
  brand : 'HP',
  RAM : '8 GB',
  clockspeed : "2 GHz"
};

// Objects can also have 'functions' called methods
let computer3 = {
  brand : 'HP',
  RAM : '8 GB',
  clockspeed : "2 GHz",
  
  printRam() {
    console.log(this.RAM)
  }
}

//this
var me = new Developer('Robin', 'Wieruch');

// '.call()' can be used to explicitly bind a function to an object
printName.call(me);

// printName() is not bound to an object so 'this' is undefined
printName();

var printInfo = function(lang1, lang2, lang3) {
  console.log(`My name is ${this.firstname} ${this.lastname} and I know ${lang1}, ${lang2}, and ${lang3}`);
}

// Create an array of languages
languages = ['Javascript','C++', 'Python'];

// Pass each argument individually by indexing the array
printInfo.call(me, languages[0], languages[1], languages[2]);

// Pass all the arguments in one array to .apply()
printInfo.apply(me, languages);

// Here we bind the me object to the printName() function and get a new function called newPrintName()
const newPrintName = printName.bind(me);

// bound newPrintName() prints appropriately
newPrintName();

// unbound printName() prints undefined
printName();

let me = {
  firstname: "Robin",
  getName: function(){
    console.log(this.name);
  }
}

// You have to bind the function to the object because just assigning it to a var
// ... is equivalent to assigning a standalone function to a var
var getMyName = me.getName.bind(me);
getMyName();

//Inheritance
class ReactDeveloper extends Developer {
  getJob() {
    return 'React Developer';
  }
}

//import export
import React from 'react';
import { firstname as username } from './myfile.js';

export default class App extends React.Component {
  render() {
    return (
      <p>Hello, {username}!</p>
    );
  }
}


//Named & Default Exports
//myfile.js
const firstname = 'Robin';
const lastname = 'Wieruch';

const person = {
  firstname,
  lastname,
};

export {
  firstname,
  lastname,
};

export default person;
//app.js
import React from 'react';
import developer, { firstname, lastname } from './myfile.js';

export default class App extends React.Component {
  render() {
    return (
      <p>Hello, {firstname}!</p>
    );
  }
}

//Library
import React, { Component } from 'react';
import axios from 'axios';

class App extends Component {
  constructor(props){
    super(props);
    this.state = {
      data: null,
    };
  }

  componentDidMount() {
    axios.get('https://api.mydomain.com')
      .then(data => this.setState({ data }));
  }

  render() {
    // JSX
  }
}

export default App;

//Components
import React from 'react'; // import the React library
require('./style.css');

import ReactDOM from 'react-dom';
import App from './app.js';

ReactDOM.render(
  <App />, // Use the component
  document.getElementById('root')
);

//Props 
//app.js
import React from 'react';

export default class App extends React.Component {
   getGreeting() {
    return this.props.greeting;
  }
   getOtherGreeting() {
    return this.props.anotherGreeting;
  }

  render() {
    return (
      <div>
        <h1> {this.getGreeting()} </h1>
        <h2> {this.getOtherGreeting()} </h2>
       </div>
    );
  }
}
//index.js
import React from 'react';
require('./style.css');

import ReactDOM from 'react-dom';
import App from './app.js';

ReactDOM.render(
  <App greeting = "greeting #1" anotherGreeting = "greeting #2"/>,
  document.getElementById('root')
);

//State
import React from 'react';

export default class App extends React.Component {
  constructor(props) {
  super(props);
    
  // creating state
  this.state = {
    firstname: 'Robin',
    lastname: 'Wieruch',
   }
  }

  getName() {
    // using state to get information
    return (`${this.state.firstname} ${this.state.lastname}`)
  }
  
  render() {
    var changeName = setTimeout(function(){
      this.setState({lastname: 'Hood'}); // setting state after delay of 5 seconds
    }.bind(this), 5000);
    return (
      <h1>{this.getName()}</h1>
    );
  }
}

//Class
//Tedious implementation
import React from 'react';

export default class Counter extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      counter: 0,
    };

    this.onIncrement = this.onIncrement.bind(this);
    this.onDecrement = this.onDecrement.bind(this);
  }

  onIncrement() {
    this.setState(state => ({ counter: state.counter + 1 }));
  }

  onDecrement() {
    this.setState(state => ({ counter: state.counter - 1 }));
  }  
  
  render() {
    return (
      <div>
        <p>{this.state.counter}</p>

        <button onClick={this.onIncrement} type="button">Increment</button>
        <button onClick={this.onDecrement} type="button">Decrement</button>
      </div>

    );
  }
}

//Shorthand
import React from 'react';

export default class Counter extends React.Component {
  state = {
    counter: 0,
  };

  onIncrement = () => {
    this.setState(state => ({ counter: state.counter + 1 }));
  }

  onDecrement = () => {
    this.setState(state => ({ counter: state.counter - 1 }));
  }


  
  render() {
    return (
      <div>
        <p>{this.state.counter}</p>

        <button onClick={this.onIncrement} type="button">Increment</button>
        <button onClick={this.onDecrement} type="button">Decrement</button>
      </div>

    );
  }
}



//Functional Stateless Components
// functional stateless components can also receive props
import React from 'react';

export function Greeting(props) {
  return(<p>{props.greeting}</p>);
}

import React from 'react';

// JavaScript ES6 arrow function without body and implicit return
export const Greeting = (props) => 
	<p>{props.greeting}</p>;

// react-create-app skeleton code
import React, { Component } from 'react';
import logo from './logo.svg';
import './App.css';

class App extends Component {
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to React</h1>
        </header>
        <p className="App-intro">
          To get started, edit <code>src/App.js</code> and save to reload.
        </p>
      </div>
    );
  }
}

export default App;

//var vs let
var guessMe = 2;
console.log("guessMe: "+guessMe);// A: guessMe is 2
( function() {
    console.log("guessMe: "+guessMe);// B: guessMe is undefined
    var guessMe = 5;
    console.log("guessMe: "+guessMe);// C: guessMe is 5
} )();
console.log("guessMe: "+guessMe);// D: guessMe is 2

let guessMe = 1;
console.log( 'guessMe: ', guessMe );// A: guessMe is 1
{
    // Temporal Dead Zone of guessMe
    //console.log( 'guessMe: ', guessMe ); <- This would give an error
    let guessMe = 2;
    console.log( 'guessMe: ', guessMe );// C: guessMe is 2
}
console.log( 'guessMe: ', guessMe );// D: guessMe is 1

//const
// temporal dead zone of PI
//PI cannot be accessed here
const PI = 3.1415;
// PI is 3.1415 and its value is final
//PI can be accessed here

//fat arrow functions
var square = a => a * a;
console.log(square(2));

//context binding, use this from outside
var Ball = function( x, y, vx, vy ) {
  this.x = x;
  this.y = y;
  this.vx = vx;
  this.vy = vy;
  this.dt = 25; // 1000/25 = 40 frames per second
  setInterval( () => { 
      this.x += vx;  
      this.y += vy;
      console.log( this.x, this.y );
  }, this.dt );
}

b = new Ball( 0, 0, 1, 1 );

// Defaults
function addCalendarEntry( 
  event, 
  date = new Date().getTime(), 
  len = 60, 
  timeout = 1000 ) {

  return len;
}
var add=addCalendarEntry( 'meeting' );
console.log(add); //outputs the default value set earlier

//Inheritance
class Shape {
  constructor( color ) {
      this.color = color;
  }

  getColor() {
      return this.color;
  }
}

class Rectangle extends Shape {
  constructor( color, width, height ) {
      super( color );
      this.width = width;
      this.height = height;
  }

  getArea() {
      return this.width * this.height;
  }
}

let rectangle = new Rectangle( 'red', 5, 8 );
console.log( "Area:\t\t" + rectangle.getArea() );
console.log( "Color:\t\t" + rectangle.getColor() );
console.log( "toString:\t" + rectangle.toString() );

//Shadowing
class User {
  constructor() {
      this.accessMatrix = {};
  }
  hasAccess( page ) {
      return this.accessMatrix[ page ];
  }
}

class SuperUser extends User {
  hasAccess( page ) {
      return true;
  }
}

var su = new SuperUser();
su.hasAccess( 'ADMIN_DASHBOARD' );

//getters setters
class Square {
  constructor( width ) { this.width = width; }
  get height() {
      console.log( 'get height' );
      return this.width;
  }
  set height( h ) {
      console.log( 'set height', h );
      this.width = h;
  }
  get area() { 
      console.log( 'get area' );
      return this.width * this.height;
  }    
}

//static methods
class C {
  static create() { return new C(); }
  constructor()   { console.log( 'Accessing constructor from the class'); }
}

var c = C.create();
//> constructor
c.create(); //this will give an error

//GAME 
class Character {
  constructor(id, name, x, y) {
      this.id = id;
      this.name = name;
      this.x = x;
      this.y = y;
      // initially the character will face right
      this.dx = 1;
      this.dy = 0;
  }
  get position() {
      return {
          x: this.x,
          y: this.y
      };
  }
  move() {
      this.x += this.dx;
      this.y += this.dy;
      if (this.x < 0) this.x = 0;
      if (this.x > 9) this.x = 9;
      if (this.y < 0) this.y = 0;
      if (this.y > 9) this.y = 9;
  }
  logPosition() {
      console.log(this.name, this.position);
  }
  collidesWith(character) {
      return character.position.x === this.x &&
          character.position.y === this.y;
  }
}
class PlayerCharacter extends Character {
  constructor(id, name, x, y) {
      super(id, name, x, y);
      this.score = 0;
  }
  faceUp() {
      this.dx = 0;
      this.dy = -1;
  }
  faceDown() {
      this.dx = 0;
      this.dy = 1;
  }
  faceLeft() {
      this.dx = -1;
      this.dy = 0;
  }
  faceRight() {
      this.dx = 1;
      this.dy = 0;
  }
  increaseScore(points) {
      this.score += points;
  }
}
class NonPlayerCharacter extends Character {
  faceRandom() {
      let dir = Math.floor(Math.random() * 4);
      this.dx = [0, 0, -1, 1][dir];
      this.dy = [-1, 1, 0, 0][dir];
  }
}

function createPlayer(id, name) {
  let x = Math.floor(Math.random() * 10),
      y = Math.floor(Math.random() * 10);
  return new PlayerCharacter(id, name, x, y);
}

function createNonPlayer(id, name) {
  let x = Math.floor(Math.random() * 10),
      y = Math.floor(Math.random() * 10);
  return new NonPlayerCharacter(id, name, x, y);
}
let npcArray = '23456'.split('').map(i => {
  return createNonPlayer(i, 'Wumpus_' + i)
});
let player = createPlayer(1, 'Hero');

function gameLoop() {
  function changeNpcDirections() {
      npcArray.forEach(npc => {
          npc.faceRandom();
      });
  }

  function moveCharacters() {
      player.move();
      npcArray.forEach(npc => {
          npc.move();
      });
  }

  function logPositions() {
      player.logPosition();
      npcArray.forEach(npc => {
          npc.logPosition();
      });
  }

  function processCollisions() {
      let len = npcArray.length;
      npcArray = npcArray.filter(
          npc => !npc.collidesWith(player));
      player.increaseScore(len - npcArray.length);
  }
  console.log('move starts');
  changeNpcDirections();
  moveCharacters();
  logPositions();
  processCollisions();
}
setInterval(gameLoop, 5000);
// influence the movement of the player by executing
// player.faceUp()
// player.faceDown()
// player.faceLeft()
// player.faceRight()

//Object Property - Shorthand Notation
//ES5
var file = {
  language: language,
  extension: extension,
  fileName: fileName
};
//ES6
var file = { language, extension, fileName };

//Destructuring
let user = {
  name        : 'Ashley',
  email       : 'ashley@ilovees2015.net',
  lessonsSeen : [ 2, 5, 6, 7, 9 ],
  nextLesson  : 10
};

let { email, nextLesson } = user;
console.log(user);
// email becomes 'ashley@ilovees2015.net'
// nextLesson becomes 10

//Rest
( (...args) => // using rest parameters
 { 
    console.log( args );
 }
) ( 1, 'Second', 3 );

//Spread
let spreadingStrings = 'Spreading Strings';
let charArray = [ ...spreadingStrings ];

//Destructuring with the Spread Operator
let notgood = 'not good'.split( '' );
let [ ,,,, ...good ] = notgood;

console.log( good );
// ["g", "o", "o", "d"]

//10*10 matrix
let nullVector = () => new Array( 10 ).fill( null );
let nullArray = nullVector().map( nullVector );

//LCS
maxCommon = ([head1,...tail1], [head2,...tail2], len = 0) => { if ( typeof head1 === 'undefined' ||
typeof head2 === 'undefined' ) { return len;
}
if ( head1 === head2 ) return maxCommon( tail1, tail2, len+1 ); let firstBranch = maxCommon( tail1, [head2, ...tail2], 0 ); let secondBranch = maxCommon([head1,...tail1], tail2, 0 ); return Math.max( ...[len, firstBranch, secondBranch ] );
}
/*******
Explanation:
We will use an optional len argument to store the number of character matches before the current iteration of maxCommon was called.
We will use recursion to process the strings.
If any of the strings have a length of 0, either head1, or head2 becomes undefined. This is our exit condition for the recursion, and we return len, i.e. the number of matching characters right before one of the strings became empty.
If both strings are non-empty, and the heads match, we recursively call maxCommon on the tails of the strings, and increase the length of the counter of the preceding common substring sequence by 1.
If the heads don’t match, we remove one character from either the first string or from the second string, and calculate their maxCommon score, with len initialized to 0 again. The longest string may either be in one of these branches, or it is equal to len, counting the matches preceding the current strings [head1,...tail1] and [head2,...tail2].
*******/


//mixins 
let View = { ... };
let ValidationMixin = { ... };
let PreloaderAnimationMixin = { ... };

let ValidatingMixinWithPreloader = Object.assign( 
    {}, 
    View, 
    ValidationMixin, 
    PreloaderAnimationMixin
);

// Concise method syntax
let shapeName = 'Rectangle', a = 5, b = 3;

let shape = { 
  shapeName, 
  a, 
  b, 
  logArea() { console.log( 'Area: ' + (a*b) ); },
  id: 0 
};

shape.logArea();

// Object Prototype Extensions
let proto = {
  whoami() { console.log('I am proto'); } 
};

let obj = { 
    whoami: function() { 
        super.whoami();
        console.log('I am obj'); 
    } 
};

Object.setPrototypeOf( obj, proto );

obj.whoami();

//Object defaults values
let newBaskets = baskets.map( item => Object.assign(
  { firstName: '-', basketValue: 0 },
  item
) );  

// Prototype
//basketProto object given
let basketProto = { 
  value: 0,
  addToBasket( itemValue ){
    this.value += itemValue;
  },
  clearBasket() {
    this.value = 0;
  },
  getBasketValue(){
    return this.value;
  },
  pay() {
    console.log( this.getBasketValue() + ' has been paid' ); 
  }

};

let myBasket = {
  items: [],
  addToBasket( itemName, itemPrice )
  {
    this.items.push( { itemName, itemPrice } );
    super.addToBasket( itemPrice );
  },
  clearBasket()
  {
    this.items = [];
    super.clearBasket();
  },
  removeFromBasket( index )
  {
    if ( typeof index !== 'number' || index < 0 || index >= this.items.length ) return;
    let removedElement = this.items.splice( index, 1 )[0];
    super.addToBasket( -removedElement.itemPrice );
	}
};
Object.setPrototypeOf( myBasket, basketProto );

//Tail call
function fib(n, a, b){
  if (n === 0) {
    return b;
  } else {
    return fib(n-1, a + b, a);
  }
};

//Name property
let guessMyName = function fName() {};
let fName2 = function() {};
let guessMyProperty = {
    prop: 1,
    methodName() {},
    get myProperty() {
        return this.prop;
    },
    set myProperty( prop ) {
        this.prop = prop;
    }
};


console.log( guessMyName.name );
//> "fName"
console.log( fName2.name );
//> "fName2"
console.log( guessMyProperty.methodName.name  );
//> "methodName"
console.log( guessMyProperty.methodName.bind( this ).name  );
//> "bound methodName"

//new.target
function MyConstructor() {
  console.log( new.target === MyConstructor, typeof new.target );
  if ( typeof new.target === 'function' ) {
      console.log( new.target.name );
  }
}

new MyConstructor(); 
//> true "function"
//> MyConstructor

MyConstructor();
//> false "undefined"

//Stack
class Stack {
	constructor() {
		this._elements = [];
	}
	get len() {
		return this._elements.length;
	}
	push( element ) {
		this._elements.push( element );
	}
	pop() {
		return this._elements.pop();
	}
}

// Class to prototype
class AbstractUser {
  constructor() {
      if ( new.target === AbstractUser ) {
          throw new Error( 'Abstract class.' );
      }
      this.accessMatrix = {};
  }
  hasAccess( page ) {
      return this.accessMatrix[ page ];
  }
}

class SuperUser extends AbstractUser {
  hasAccess( page ) {
      return true;
  }
}

let su = new SuperUser();

//let au = new AbstractUser();
// ^ Throws the new error

let AbstractUser = function()
{
	if ( new.target === AbstractUser )
  {
		throw new Error( 'Abstract class' );
  }
	this.accessMatrix = {}; 
};

AbstractUser.prototype.hasAccess = function( page )
{
  return this.accessMatrix[ page ];
};

let SuperUser = function()
{
  AbstractUser.call( this );
};

SuperUser.prototype = Object.create( AbstractUser.prototype );
SuperUser.prototype.constructor = SuperUser;
SuperUser.prototype.hasAccess = function( page )
{
  return true;
};

//let su = new SuperUser();
//let au = new AbstractUser();
// The above two lines of code will give an error: Uncaught Error: Abstract class cannot be instantiated.(...)


//Symbol
let leftNode = Symbol( 'Binary tree node' );
let rightNode = Symbol( 'Binary tree node' );
console.log( leftNode === rightNode );
//> false

let privateProperty1 = Symbol.for( 'firstName' );
let privateProperty2 = Symbol.for( 'firstName' );

myObject[ privateProperty1 ] = 'Dave';
myObject[ privateProperty2 ] = 'Zsolt';

console.log( myObject[ privateProperty1 ] );
// Zsolt

//Symbol as semi private property key
let Square = (function() {

  const _width = Symbol('width');

  return class {
      constructor( width0 ) {
          this[_width] = width0;
      }
      getWidth() {
          return this[_width];
      }
  }  

} )();

//Enum
const directions = {
  UP   : Symbol( 'UP' ),
  DOWN : Symbol( 'DOWN' ),
  LEFT : Symbol( 'LEFT' ),
  RIGHT: Symbol( 'RIGHT' )
};
console.log(directions);

//simulate truly private fields in JavaScript
function F() {
  let privateProperty = 'b';
  this.publicProperty = 'a';
}

let f = new F();
console.log(f.publicProperty);  // returns 'a'
console.log(f.privateProperty); // returns undefined
//For Class
class C
{ 
  constructor()
  {
    let privateProperty = 'a'; 
     Object.assign( this,
     {
       logPrivateProperty()
       {
         console.log( privateProperty );
       }
     } );
  }
}

let c = new C();
c.logPrivateProperty();

//for-of
let message = 'hello';

for( let i in message ) {
    console.log( message[i] );
}

for( let ch of message ) {
  console.log( ch );
}

//UTF32 parse
let text = '\u{1F601}\u{1F43C}'; 
console.log( 'text: ', text );

for( let i in text ) { 
    console.log(text[i]); //bad
}; 

console.log('-----'); 

for ( let c of text ) { //good
    console.log( c ); 
};

//for-of destructuring
let flights = [
  { source: 'Dublin', destination: 'Warsaw' },
  { source: 'New York', destination: 'Phoenix' }
];

for ( let { source, destination } of flights ) {
  console.log( source, destination );
}

for ( let { source } of flights ) {
  console.log( source );
}

let divs = document.querySelectorAll( 'div' );
for ( let div of divs )
{
    let rand = Math.floor ( Math.random() * 3 );
    div.style.color = [ '#990000', '#009900', '#000099'][ rand ];
}

//Print all emojis
let prefix = '1F6';
let digits4 = '01234';
let digits5 = '01234567890ABCDEF';
let emojis = '';

for ( let digit4 of digits4 ) {
  for ( let digit5 of digits5 ) {
    let hexCode = '0x' + prefix + digit4 + digit5;
    emojis += String.fromCodePoint( hexCode );
  }
}
console.log( emojis );

//strings methods
console.log('Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz'.startsWith( 'Rindfleisch' ));
//> true
console.log('not good'.endsWith( 'good' ));
//> true

console.log('good or bad'.includes( ' or ' ));
//> true

console.log('ha'.repeat( 4 ));
//> 'hahahaha'

//Template Literals
let x = 555; //<-notice 555
let evaluatedTemplate = `${x} === 555 is ${x === 555}`;
console.log(evaluatedTemplate);
// evaluatedTemplate becomes "555 === 555 is true"

let y = '555'; //<-notice '555'
evaluatedTemplate = `${y} === 555 is ${y === 555}`;
console.log(evaluatedTemplate);
// evaluatedTemplate becomes "555 === 555 is false"

//Tagged templates
let sub1=1, sub2=2, sub3 = 3; 
( (x, ...subs) => { 
        console.log( x, ...subs ); 
    })`${sub1}abc ${sub2} def${sub3}`

let upper = (textArray, ...substitutions) => {
	let template = '';
	for ( let i = 0; i < substitutions.length; ++i ) {
		let sub = substitutions[ i ];
		template += textArray[ i ];
		template += typeof sub === 'string' ?
		sub.toUpperCase() : sub;
	}
	template += textArray[ textArray.length - 1 ];
	return template;
};

let a = 1, b = 'ab', c = 'DeF';
console.log(upper`x ${a} x ${b} x ${c} x`);

//Set
let colors = new Set();

colors.add( 'red' );
colors.add( 'green' );
colors.add( 'red' );   // duplicate elements are added only once
console.log( colors );
//> Set {"red", "green"}

console.log( 'Size: ' + colors.size );
//> 2

console.log( 'has green: ' + colors.has( 'green' ) + '\nhas blue: ' + colors.has( 'blue' ) );
//> true false
colors.delete( 'green' )
//> true
colors.delete( 'green' )
//> false

console.log('forEach function:')
moreColors.forEach( value => { console.log( value ) } );
//> red
//> blue

console.log('\nfor...of loop:')
for ( let value of moreColors ) {
    console.log( value );
}
//> red
//> blue

console.log('\nspread operator:')
console.log( [...moreColors] );
//> ["red", "blue"]

//Maps
let horses = new Map();

horses.set( 8, 'Chocolate' );
horses.set( 3, 'Filippone' );
console.log(horses);

let horses = new Map( [[8, 'Chocolate'], [3, 'Filippone' ]] );
console.log(horses);

console.log('Size:\t\t'+horses.size);
//> 2

console.log('has id=3:\t'+horses.has( 3 ));
//> true

console.log('value at key=3:\t'+horses.get( 3 ));
//> "Filippone"

horses.delete( 3 );
console.log('\nAfter deleting:\t');
console.log(horses);
//> true

console.log('forEach function:');
horses.forEach( ( value, key ) => { console.log( value, key ) } );
//> Chocolate 8
//> Filippone 3

console.log('\nfor...of loop:');
for ( let [ key, value ] of horses ) {
    console.log( key, value );
}
//> 8 "Chocolate"
//> 3 "Filippone"

console.log('\nspread operators:');
console.log( [...horses] );
//> [[8,"Chocolate"],[3,"Filippone"]]

//weak set
let firstElement = { order: 1 }, secondElement = { order: 2 };
let ws = new WeakSet( [ firstElement, secondElement ] );

console.log('has firstElement: '+ws.has( firstElement ));
//> true

delete firstElement;
// firstElement is removed from the weak set

//weak map
let firstElement = { order: 1 }, secondElement = { order: 2 };
let wm = new WeakMap();

wm.set( firstElement, 1 );
wm.set( secondElement, {} );

console.log(wm.get( secondElement ));
//> {}
 
delete secondElement;
// secondElement is removed from the weak map

let Square = ( function() {
  let _width = new WeakMap();
  class Square {
    constructor( width ) {
    _width.set( this, width );
  }
  get area() {
    let width = _width.get( this );
    return width * width;
  }
}
return Square;
} )();

//Iterables
let countdownIterator = {
  countdown: 10,
  next() {
      this.countdown -= 1;
      return {
          done: this.countdown === 0,
          value: this.countdown
      };
  }    
};  

let countdownIterable = {
  [Symbol.iterator]() {
      return Object.assign( {}, countdownIterator ) 
  }
};

let iterator = countdownIterable[Symbol.iterator]();

console.log(iterator.next());
//> Object {done: false, value: 9}

console.log(iterator.next());
//> Object {done: false, value: 8}


for ( let element of iterableObject ) {
  console.log( element );
}

console.log( [...iterableObject] );

console.log([...countdownIterable]);
//> [9, 8, 7, 6, 5, 4, 3, 2, 1]

//Iterables with Sets and Maps
let colors = new Set( [ 'red', 'yellow', 'green' ] );
let horses = new Map( [
    [5, 'QuickBucks'], 
    [8, 'Chocolate'], 
    [3, 'Filippone']
] );

console.log( colors.entries() );
//> SetIterator {["red", "red"], ["yellow", "yellow"], ["green", "green"]}

console.log('\n')
console.log( colors.keys() );
//> SetIterator {"red", "yellow", "green"}

console.log( colors.values() );
//> SetIterator {"red", "yellow", "green"}

console.log( horses.entries() );
//> MapIterator {[5, "QuickBucks"], [8, "Chocolate"], [3, "Filippone"]}

console.log( horses.keys() );
//> MapIterator {5, 8, 3}

console.log( horses.values() );
//> MapIterator {"QuickBucks", "Chocolate", "Filippone"}

for ( let [key, value] of horses ) {
  console.log( key, value );
}

///Generator
function *getLampIterator() {
  yield 'red';
  yield 'green';
  return 'lastValue';
  // implicit: return undefined;
}

let lampIterator = getLampIterator();

console.log( lampIterator.next() );
//> {value: "red", done: false}

console.log( lampIterator.next() );
//> {value: "green", done: false}

console.log( lampIterator.next() );
//> {value: "lastValue", done: true}

function *getLampIterator() {
  yield 'red';
  yield 'green';
  return 'lastValue';
  // implicit: return undefined;
}

let lampIterator = getLampIterator();

console.log( lampIterator.next() );
//> {value: 'red', done: false}

console.log( [...lampIterator] );
//> ['green']

//Combining Generators
let countdownGenerator = function *() {
  let i = 10;
  while ( i > 0 ) yield --i;
}

let lampGenerator = function *() {
  yield 'red';
  yield 'green';
}

let countdownThenLampGenerator = function *() {
  yield *countdownGenerator();
  yield *lampGenerator();
}

console.log( [...countdownThenLampGenerator()] );
//>>[ 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 'red', 'green' ]

let greetings = function *() {
  let name = yield 'Hi!';
  yield `Hello, ${ name }!`;
}

let greetingIterator = greetings();

console.log( greetingIterator.next() );
//> Object {value: "Hi!", done: false}

console.log( greetingIterator.next( 'Lewis' ) );
//> Object {value: "Hello, Lewis!", done: false}

//Lazy evaluation
function *filter( iterable, filterFunction ) {
  for( let element of iterable ) {
      if ( filterFunction( element ) ) yield element;
  }
}
let evenFibonacci = filter( fibonacci(), x => x%2 === 0 );

console.log(evenFibonacci.next());
//> {value: 0, done: false}
console.log(evenFibonacci.next());
//> {value: 2, done: false}
console.log(evenFibonacci.next());
//> {value: 8, done: false}
console.log(evenFibonacci.next());
//> {value: 34, done: false}
console.log(evenFibonacci.next());
//> {value: 144, done: false}


/*
Lazy evaluation is essential when we work on a large set of data. For instance,
if you have 1000 accounts, chances are that you don’t want to transform all
of them if you just want to render the first ten on screen. This is when lazy
evaluation comes into play.
*/

//Promise
let promise1 = new Promise( function( resolve, reject ) {
  // call resolve( value ) to resolve a promise
  // call reject( reason ) to reject a promise
} );

// Create a resolved promise
let promise2 = Promise.resolve( 5 );
console.log(promise1)
console.log(promise2)

var p = Promise.resolve( 5 );

p.then( ( value ) => console.log( 'Value:', value ) )
 .then( () => { throw new Error('Error in second handler' ) } )
 .catch( ( error ) => console.log(error.toString() ) );

 //Promise all
 var loan1 = new Promise( (resolve, reject) => { 
  setTimeout( () => resolve( 110 ) , 1000 ); 
}); 
var loan2 = new Promise((resolve, reject) => { 
  setTimeout( () => resolve( 120 ) , 2000 ); 
});
var loan3 = new Promise( (resolve, reject) => {
  reject( 'Bankrupt' );
});

Promise.all([ loan1, loan2, loan3 ]).then( value => { 
  console.log(value);
}, reason => {
  console.log(reason);
} );

//Small example
<input type="text" class="js-textfield" />
<button class="js-button">
    Enable the Textfield
</button>

document.body.innerHTML = `
<input type="text" class="js-textfield" />
<button class="js-button">
  Enable the Textfield
</button>` +
document.body.innerHTML;

let enableFields = function() {
  document.querySelector( '.js-textfield' )
    .removeAttribute( 'disabled' );
  document.querySelector( '.js-button' )
    .removeAttribute( 'disabled' );
}
let disableFields = function() {
  document.querySelector( '.js-textfield' )
    .setAttribute( 'disabled', true );
  document.querySelector( '.js-button' )
    .setAttribute( 'disabled', true );
}

let parse = function( response ) {
  let element = document.querySelector( '.js-names' );
  
  element.innerHTML =
    JSON.parse( response )
      .map( element => element.name )
        .join(',');
  enableFields();
}
let errorHandler = function() {
  console.log( 'error' );
  enableFields();
}
new Promise( function( resolve, reject ) {
  disableFields();
  let request = new XMLHttpRequest();
  request.onreadystatechange = function() {
    if ( this.status === 200 && this.readyState === 4 ) {
      resolve( this.response );
    }
  }
  request.onerror = function() {
    reject( new Error( this.statusText ) );
  }
  request.open(
    'GET',
      'http://erroneousurl.com/users'
  );
  request.send();
} ).then( parse )
  .catch( errorHandler );


//Reflect
let target = function getArea( width, height ) {
  return `${ width * height }${this.units}`;
}
let thisValue = { units: 'cm' };
let args = [ 5, 3 ];


console.log('Area: '+Reflect.apply( target, thisValue, args ));

let target = class Account {
  constructor( name, email ) {
      this.name = name;
      this.email = email;
  }  
  get contact() {
      return `${this.name} <${this.email}>`;
  }
};
let args = [ 
  'Zsolt', 
  'info@zsoltnagy.eu' 
];

let myAccount = Reflect.construct(
  target,
  args );

console.log(myAccount.contact); 
//> "Zsolt <info@zsoltnagy.eu>"

let classOfMyAccount = Reflect.getPrototypeOf( myAccount );

console.log( classOfMyAccount.prototype === myAccount.prototype );

let newProto = {
  get contact() {
      return `${this.name} - 555-1269`;
  }
}

Reflect.setPrototypeOf( myAccount, newProto );

console.log( myAccount.contact ); 
//> "Zsolt - 555-1269"

let target = class Account {
  constructor( name, email ) {
      this.name = name;
      this.email = email;
  }  
  get contact() {
      return `${this.name} <${this.email}>`;
  }
};
let args = [ 
  'Zsolt', 
  'info@zsoltnagy.eu' 
];

let myAccount = Reflect.construct(
  target,
  args );


console.log(Reflect.has( myAccount, 'name' ));
console.log(Reflect.has( myAccount, 'contact' ));

console.log(Reflect.ownKeys( myAccount ));

console.log(Reflect.get( myAccount, 'name' ));
//> "Zsolt"

let target = myAccount;
let property = 'age';
let newValue = 32;

Reflect.set(
    myAccount,
    property,
    newValue
);

console.log(myAccount.age );
//> 32

let target = {};
let key = 'response';
let attributes = {
    value: 200,
    writable: true,
    enumerable: true
};

Reflect.defineProperty( 
    target, 
    key, 
    attributes
);

//Proxy
class Student {
  constructor(first, last, scores) {
      this.firstName = first;
      this.lastName = last;
      this.testScores = scores;
  }
  get average() {
      let average = this.testScores.reduce( 
          (a,b) => a + b, 
          0 
      ) / this.testScores.length;
      return average;
  }
}

let john = new Student( 'John', 'Dwan', [60, 80, 80] );
console.log(john);

let johnMethodProxy = new Proxy( john, {
  get: function( target, key, context ) {
      if ( key === 'average' ) {
          return target.average;
      }
  } 
});

console.log(johnMethodProxy.firstName);
//undefined
console.log(johnMethodProxy.average);
//73.33333333333333

///Proxy and Reflect
let factorial = n =>
    n <= 1 ? n : n * factorial( n - 1 );

let numOfCalls = 0;
factorial = new Proxy( factorial, {
   apply: function( target, thisValue, args ) {
        numOfCalls += 1;
        return Reflect.apply(
            target, 
            thisValue, 
            args 
        );
   } 
});

console.log(factorial( 5 ) && numOfCalls);
//> 5

let payload = {
  website: 'zsoltnagy.eu',
  article: 'Proxies in Practice',
  viewCount: 15496
}

let revocable = Proxy.revocable( payload, {
 get: function( ...args ) {
      console.log( 'Proxy' );
      return Reflect.get( ...args );
 } 
});

let proxy = revocable.proxy;

console.log(proxy.website);
//> Proxy
//> "zsoltnagy.eu"

revocable.revoke();

proxy.website;
//> Uncaught TypeError: Cannot perform 'get' on a proxy that 
//> has been revoked
//>    at <anonymous>:3:6

//Short hand
// Create a revocable proxy
let {proxy, revoke} = Proxy.revocable( payload, {
  get: function( ...args ) {
       console.log( 'Proxy' );
       return Reflect.get( ...args );
  } 
});

// Revoke the proxy
revoke();

//Number Extensions
console.log(Number.isInteger( 5 ));

let maxVal = Number.MAX_SAFE_INTEGER;
let minVal = Number.MIN_SAFE_INTEGER;

console.log(Number.isSafeInteger( maxVal ));
//> true

console.log(Number.isSafeInteger( maxVal + 1 ));
//> false

// base 10 = decimal input (default):
console.log(Number.parseInt( '1234', 10 ));
//> 1234

// base 16 = hexadecimal input:
console.log(Number.parseInt( 'ff', 16 ));  
//> 255

console.log(Number.parseFloat( '1.2' ));
//> 1.2

console.log(Number.isFinite( 5 ));
//> true

console.log(Number.isNaN( 0/0 ));
//> true

//ES2016
let base = 10;
let exponent = 3;

console.log(base ** exponent);

console.log([1, 2, 3].includes( 2 ));
// check starts from the 2nd element
console.log([1, 2, 3].includes( 2, 1 ));  

// check starts from the 3rd element
console.log([1, 2, 3].includes( 2, 2 )); 

console.log('JavaScript'.includes( 'Java' ));

//ES2017
let account = {
  first: 'Zsolt',
  last: 'Nagy',
  email: 'info@zsoltnagy.eu'
};

console.log(Object.keys( account ));
console.log(Object.values( account ));
console.log(Object.entries( account ));

let iterator = Object.values( account ).entries();
console.log(iterator);
//> ArrayIterator {}

console.log( iterator.next() );
//> { value: [0, "Zsolt"], done: false }

for ( let [val, key] of iterator ) {
    console.log( val, key );
}
//> 1 "Nagy"
//> 2 "info@zsoltnagy.eu"

let player = {
  cards: [ 'Ah', 'Qc' ], 
  chips: 1000 
};

let descriptors = 
  Object.getOwnPropertyDescriptors( player );

console.log( descriptors );
console.log();
console.log( descriptors.cards );

let amounts = [
  '1234.0',
  '1',
  '2.56'
];

console.log( `|dddddd.ff|` );
for ( let amount of amounts ) {
  let [ front, back = '' ] = amount.split('.');
  front = front.padStart( 6 );
  back = back.padEnd( 2, '0' );
  console.log( `|${front}.${back}|` );
}

const loadData = async () => {
  const [resultSet1, resultSet2] = await Promise.all([
      asyncQuery1().then( displayResultSet1 ),
      asyncQuery2().then( displayResultSet2 )
  ] );
}

const loadData = () => {
  asyncQuery1().then( displayResultSet1 );
  asyncQuery2().then( displayResultSet2 );
}

/// *** Fundamentals ***
//Components
import React from 'react';

function App() {
  // do something in between
  return (
    <div>
      <h1>Hello World</h1>
    </div>
  );
}

export default App;

//List
import React from 'react';

const list = [
  {
    title: 'React',
    url: 'https://reactjs.org/',
    author: 'Jordan Walke',
    num_comments: 3,
    points: 4,
    objectID: 0,
  },
  {
    title: 'Redux',
    url: 'https://redux.js.org/',
    author: 'Dan Abramov, Andrew Clark',
    num_comments: 2,
    points: 5,
    objectID: 1,
  },
];

function App() {
  return (
    <div>
      <h1>My Hacker Stories</h1>

      <label htmlFor="search">Search: </label>
      <input id="search" type="text" />

      <hr />

      {list.map(function(item) {
        return (
          <div key={item.objectID}>
            <span>
              <a href={item.url}>{item.title}</a>
            </span>
            <span>{item.author}</span>
            <span>{item.num_comments}</span>
            <span>{item.points}</span>
          </div>
        );
      })}
    </div>
  );
}

export default App;

//Arrow
// function declaration
function () { ... }

// arrow function declaration
const () => { ... }

// allowed
const item => { ... } 

// allowed
const (item) => { ... }

// not allowed
const item, index => { ... }

// allowed
const (item, index) => { ... }

// with block body
count => {
  // perform any task in between

  return count + 1;
}

// with concise body
count =>
  count + 1;

//Handler
// don't do this
<input
  id="search"
  type="text"
  onChange={handleChange()}
/>

// do this instead
<input
  id="search"
  type="text"
  onChange={handleChange}
/>

const handleChange = event => {
  console.log(event.target.value);
};

//props
const App = () => {
  const stories = [ ... ];

  const handleChange = event => { ... };

  return (
    <div>
      <h1>My Hacker Stories</h1>

      <label htmlFor="search">Search: </label>
      <input id="search" type="text" onChange={handleChange} />

      <hr />


      <List list={stories} />
    </div>
  );
};

const List = props =>
  props.list.map(item => (
    <div key={item.objectID}>
      <span>
        <a href={item.url}>{item.title}</a>
      </span>
      <span>{item.author}</span>
      <span>{item.num_comments}</span>
      <span>{item.points}</span>
    </div>
  ));

//destructuring
// basic array definition
const list = ['a', 'b'];

// no array destructuring
const itemOne = list[0];
const itemTwo = list[1];

// array destructuring
const [firstItem, secondItem] = list;

//states
const App = () => {
  const stories = [ ... ];

  const [searchTerm, setSearchTerm] = React.useState('');

  const handleChange = event => {

    setSearchTerm(event.target.value);

  };

  return (
    <div>
      <h1>My Hacker Stories</h1>

      <label htmlFor="search">Search: </label>
      <input id="search" type="text" onChange={handleChange} />

      <p>
        Searching for <strong>{searchTerm}</strong>.
      </p>
      <hr />

      <List list={stories} />
    </div>
  );
};

//callback handler
const App = () => {
  const stories = [ ... ];
 
  const [searchTerm, setSearchTerm] = React.useState('');
 
  const handleSearch = event => {
    setSearchTerm(event.target.value);
  };
 
  const searchedStories = stories.filter(story =>
    story.title.toLowerCase().includes(searchTerm.toLowerCase())
  );
 
  return (
    <div>
      <h1>My Hacker Stories</h1>
 
      <Search onSearch={handleSearch} />
 
      <hr />
 
      <List list={searchedStories} />
    </div>
  );
 };

 const Search = props => (
  <div>
    <label htmlFor="search">Search: </label>
 
    <input id="search" type="text" onChange={props.onSearch} />
  </div>
 );

//Props Handling
const Search = ({ search, onSearch }) => (

  <div>
    <label htmlFor="search">Search: </label>
    <input
      id="search"
      type="text"
 
      value={search}
      onChange={onSearch}
    />
  </div>
 );

const List = ({ list }) =>
list.map(item => <Item key={item.objectID} item={item} />);

const Item = ({ item }) => (
 <div>
   <span>
     <a href={item.url}>{item.title}</a>
   </span>
   <span>{item.author}</span>
   <span>{item.num_comments}</span>
   <span>{item.points}</span>
 </div>
);

//side effects handling
const App = () => {
  ...
 
  const [searchTerm, setSearchTerm] = React.useState(
    localStorage.getItem('search') || 'React'
  );
 
  React.useEffect(() => {
    localStorage.setItem('search', searchTerm);
  }, [searchTerm]);
 
  const handleSearch = event => {
    setSearchTerm(event.target.value);
  };
 
 
  ...
 );

//custom hooks
const useSemiPersistentState = (key, initialState) => {

  const [value, setValue] = React.useState(
 
    localStorage.getItem(key) || initialState
 
  );
 
  ...
 };
 
 const App = () => {
  ...
 
  const [searchTerm, setSearchTerm] = useSemiPersistentState(
 
    'search',
    'React'
 
  );
 
  ...
 };

 //Fragments
 const Search = ({ search, onSearch }) => (
  <>
    <label htmlFor="search">Search: </label>
    <input
      id="search"
      type="text"
      value={search}
      onChange={onSearch}
    />
  </>
);

//Reusable components
const InputWithLabel = ({
  id,
  label,
  value,
  type = 'text',
  onInputChange,
}) => (
  <>
    <label htmlFor={id}>{label}</label>
    &nbsp;
    <input
      id={id}
      type={type}
      value={value}
      onChange={onInputChange}
    />
  </>
);

//Composition
const InputWithLabel = ({
  id,
  value,
  type = 'text',
  onInputChange,

  children,

}) => (
  <>

    <label htmlFor={id}>{children}</label>

    &nbsp;
    <input
      id={id}
      type={type}
      value={value}
      onChange={onInputChange}
    />
  </>
);

const App = () => {
  ...

  return (
    <div>
      <h1>My Hacker Stories</h1>

      <InputWithLabel
        id="search"
        value={searchTerm}
        onInputChange={handleSearch}
      >

        <strong>Search:</strong>

      </InputWithLabel>

      ...
    </div>
  );
};

//Imperative
const InputWithLabel = ({
  id,
  value,
  type = 'text',
  onInputChange,
  isFocused,
  children,
}) => {
  const inputRef = React.useRef();

  React.useEffect(() => {
    if (isFocused) {
      inputRef.current.focus();
    }
  }, [isFocused]);

  return (
    <>
      <label htmlFor={id}>{children}</label>
      &nbsp;
      <input
        ref={inputRef}
        id={id}
        type={type}
        value={value}
        onChange={onInputChange}
      />
    </>
  );
};

<InputWithLabel
        id="search"
        value={searchTerm}
        isFocused
        onInputChange={handleSearch}
      >
        <strong>Search:</strong>
</InputWithLabel>

//Inline handler
const [stories, setStories] = React.useState(initialStories);

const handleRemoveStory = item => {
  const newStories = stories.filter(
    story => item.objectID !== story.objectID
  );

  setStories(newStories);
};

<List list={searchedStories} onRemoveItem={handleRemoveStory} />

const List = ({ list, onRemoveItem }) =>
  list.map(item => (
    <Item
      key={item.objectID}
      item={item}
      onRemoveItem={onRemoveItem}
    />
  ));

  const Item = ({ item, onRemoveItem }) => (
    <div>
      <span>
        <a href={item.url}>{item.title}</a>
      </span>
      <span>{item.author}</span>
      <span>{item.num_comments}</span>
      <span>{item.points}</span>
      <span>
        <button type="button" onClick={() => onRemoveItem(item)}>
          Dismiss
        </button>
      </span>
    </div>
  );

//Async Data
const App = () => {
  ...

  const [stories, setStories] = React.useState([]);
  React.useEffect(() => {
    getAsyncStories().then(result => {
      setStories(result.data.stories);
    });
  }, []);
  ...
};

const getAsyncStories = () =>
  new Promise(resolve =>
    resolve({ data: { stories: initialStories } })
  );

//Conditional Rendering
const App = () => {
  ...

  const [stories, setStories] = React.useState([]);

  const [isLoading, setIsLoading] = React.useState(false);


  React.useEffect(() => {

    setIsLoading(true);

    getAsyncStories().then(result => {
      setStories(result.data.stories);

      setIsLoading(false);

    });
  }, []);

  ...
};

const App = () => {
  ...

  return (
    <div>
      ...

      <hr />

      {isError && <p>Something went wrong ...</p>}
      {isLoading ? (
        <p>Loading ...</p>
      ) : (
        ...
      )}
    </div>
  );
};

//Reducer
const App = () => {
  ...

  const [stories, dispatchStories] = React.useReducer(
    storiesReducer,
    []
  );

  ...
};

const storiesReducer = (state, action) => {

  switch (action.type) {
    case 'SET_STORIES':
      return action.payload;
    case 'REMOVE_STORY':
      return state.filter(
        story => action.payload.objectID !== story.objectID
      );
    default:
      throw new Error();
  }

};

//Avoid impossible state
const App = () => {
  ...

  const [stories, dispatchStories] = React.useReducer(
    storiesReducer,
    []
  );
  const [isLoading, setIsLoading] = React.useState(false);
  const [isError, setIsError] = React.useState(false);

  ...
};

//into

const App = () => {
  ...

  const [stories, dispatchStories] = React.useReducer(
    storiesReducer,
    { data: [], isLoading: false, isError: false }
  );

  ...

  const searchedStories = stories.data.filter(story =>

    story.title.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div>
      ...

      {stories.isError && <p>Something went wrong ...</p>}

      {stories.isLoading ? (
        <p>Loading ...</p>
      ) : (
        <List
          list={searchedStories}
          onRemoveItem={handleRemoveStory}
        />
      )}
    </div>
  );
};

const storiesReducer = (state, action) => {
  switch (action.type) {

    case 'STORIES_FETCH_INIT':
      return {
        ...state,
        isLoading: true,
        isError: false,
      };
    case 'STORIES_FETCH_SUCCESS':
      return {
        ...state,
        isLoading: false,
        isError: false,
        data: action.payload,
      };
    case 'STORIES_FETCH_FAILURE':
      return {
        ...state,
        isLoading: false,
        isError: true,
      };
    case 'REMOVE_STORY':
      return {
        ...state,
        data: state.data.filter(
          story => action.payload.objectID !== story.objectID
        ),
      };

    default:
      throw new Error();
  }
};

//Server side
const App = () => {
  ...

  React.useEffect(() => {

    if (!searchTerm) return;

    dispatchStories({ type: 'STORIES_FETCH_INIT' });

    fetch(`${API_ENDPOINT}${searchTerm}`)
      .then(response => response.json())
      .then(result => {
        dispatchStories({
          type: 'STORIES_FETCH_SUCCESS',
          payload: result.hits,
        });
      })
      .catch(() =>
        dispatchStories({ type: 'STORIES_FETCH_FAILURE' })
      );

  }, [searchTerm]);


  ...
};

//Avoid loop, memorized
const App = () => {
  ...

  // A
  const handleFetchStories = React.useCallback(() => {
    if (!searchTerm) return;

    dispatchStories({ type: 'STORIES_FETCH_INIT' });

    fetch(`${API_ENDPOINT}${searchTerm}`)
      .then(response => response.json())
      .then(result => {
        dispatchStories({
          type: 'STORIES_FETCH_SUCCESS',
          payload: result.hits,
        });
      })
      .catch(() =>
        dispatchStories({ type: 'STORIES_FETCH_FAILURE' })
      );

  }, [searchTerm]); // E


  React.useEffect(() => {

    handleFetchStories(); // C
  }, [handleFetchStories]); // D


  ...
};

//Explicit Fetch
const App = () => {
  ...

  return (
    <div>
      <h1>My Hacker Stories</h1>

      <InputWithLabel
        id="search"
        value={searchTerm}
        isFocused

        onInputChange={handleSearchInput}

      >
        <strong>Search:</strong>
      </InputWithLabel>


      <button
        type="button"
        disabled={!searchTerm}
        onClick={handleSearchSubmit}
      >
        Submit
      </button>

      ...
    </div>
  );
};

const App = () => {
  const [searchTerm, setSearchTerm] = useSemiPersistentState(
    'search',
    'React'
  );

  const [url, setUrl] = React.useState(
    `${API_ENDPOINT}${searchTerm}`
  );

  ...
  const handleSearchInput = event => {

    setSearchTerm(event.target.value);
  };

  const handleSearchSubmit = () => {
    setUrl(`${API_ENDPOINT}${searchTerm}`);
  };

  ...
};

const App = () => {
  ...

  const handleFetchStories = React.useCallback(() => {
    dispatchStories({ type: 'STORIES_FETCH_INIT' });

    fetch(url)
      .then(response => response.json())
      .then(result => {
        dispatchStories({
          type: 'STORIES_FETCH_SUCCESS',
          payload: result.hits,
        });
      })
      .catch(() =>
        dispatchStories({ type: 'STORIES_FETCH_FAILURE' })
      );

  }, [url]);


  React.useEffect(() => {
    handleFetchStories();
  }, [handleFetchStories]);

  ...
};

//axios

npm install axios

import axios from 'axios';
const App = () => {
  ...

  const handleFetchStories = React.useCallback(() => {
    dispatchStories({ type: 'STORIES_FETCH_INIT' });

    axios
      .get(url)

      .then(result => {
        dispatchStories({
          type: 'STORIES_FETCH_SUCCESS',

          payload: result.data.hits,

        });
      })
      .catch(() =>
        dispatchStories({ type: 'STORIES_FETCH_FAILURE' })
      );
  }, [url]);

  ...
};

const App = () => {
  ...

  const handleFetchStories = React.useCallback(async () => {
    dispatchStories({ type: 'STORIES_FETCH_INIT' });

    try {
      const result = await axios.get(url);

      dispatchStories({
        type: 'STORIES_FETCH_SUCCESS',
        payload: result.data.hits,
      });
    } catch {
      dispatchStories({ type: 'STORIES_FETCH_FAILURE' });
    }

  }, [url]);

  ...
};

//Form
const SearchForm = ({
  searchTerm,
  onSearchInput,
  onSearchSubmit,
}) => (
  <form onSubmit={onSearchSubmit}>
    <InputWithLabel
      id="search"
      value={searchTerm}
      isFocused
      onInputChange={onSearchInput}
    >
      <strong>Search:</strong>
    </InputWithLabel>

    <button type="submit" disabled={!searchTerm}>
      Submit
    </button>
  </form>
);

const App = () => {
  ...

  return (
    <div>
      <h1>My Hacker Stories</h1>
      <SearchForm
        searchTerm={searchTerm}
        onSearchInput={handleSearchInput}
        onSearchSubmit={handleSearchSubmit}
      />

      <hr />

      {stories.isError && <p>Something went wrong ...</p>}

      {stories.isLoading ? (
        <p>Loading ...</p>
      ) : (
        <List list={stories.data} onRemoveItem={handleRemoveStory} />
      )}
    </div>
  );
};

//Legacy class component
class InputWithLabel extends React.Component {
  render() {
    const {
      id,
      value,
      type = 'text',
      onInputChange,
      children,
    } = this.props;

    return (
      <>
        <label htmlFor={id}>{children}</label>
        &nbsp;
        <input
          id={id}
          type={type}
          value={value}
          onChange={onInputChange}
        />
      </>
    );
  }
}

//function components
const InputWithLabel = ({
  id,
  value,
  type = 'text',
  onInputChange,
  children,
}) => (
  <>
    <label htmlFor={id}>{children}</label>
    &nbsp;
    <input
      id={id}
      type={type}
      value={value}
      onChange={onInputChange}
    />
  </>
);

//Legacy state
class App extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      searchTerm: 'React',
    };
  }

  render() {

    const { searchTerm } = this.state;
    return (
      <div>
        <h1>My Hacker Stories</h1>

        <SearchForm
          searchTerm={searchTerm}
          onSearchInput={() => this.setState({
            searchTerm: event.target.value
          })}
        />
      </div>
    );
  }
}

//CSS module
mv src/App.css src/App.module.css

import styles from './App.module.css';

const Item = ({ item, onRemoveItem }) => (

  <div className={styles.item}>
    <span style={{ width: '40%' }}>

      <a href={item.url}>{item.title}</a>
    </span>

    <span style={{ width: '30%' }}>{item.author}</span>
    <span style={{ width: '10%' }}>{item.num_comments}</span>
    <span style={{ width: '10%' }}>{item.points}</span>
    <span style={{ width: '10%' }}>

      <button
        type="button"
        onClick={() => onRemoveItem(item)}

        className={`${styles.button} ${styles.buttonSmall}`}

      >
        Dismiss
      </button>
    </span>
  </div>
);

//SVG
import { ReactComponent as Check } from './check.svg';

const Item = ({ item, onRemoveItem }) => (
  <div className="item">
    <span style={{ width: '40%' }}>
      <a href={item.url}>{item.title}</a>
    </span>
    <span style={{ width: '30%' }}>{item.author}</span>
    <span style={{ width: '10%' }}>{item.num_comments}</span>
    <span style={{ width: '10%' }}>{item.points}</span>
    <span style={{ width: '10%' }}>
      <button
        type="button"
        onClick={() => onRemoveItem(item)}
        className="button button_small"
      >

        <Check height="18px" width="18px" />

      </button>
    </span>
  </div>
);


//Optimization made up state
const useSemiPersistentState = (key, initialState) => {

  const isMounted = React.useRef(false);


  const [value, setValue] = React.useState(
    localStorage.getItem(key) || initialState
  );

  React.useEffect(() => {

    if (!isMounted.current) {
      isMounted.current = true;
    } else {

      console.log('A');
      localStorage.setItem(key, value);

    }

  }, [value, key]);

  return [value, setValue];
};

//logging in component has no function body
const App = () => {
  ...

  console.log('B:App');
  return ( ... );
};

const List = ({ list, onRemoveItem }) =>

  console.log('B:List') || //here

  list.map(item => (
    <Item
      key={item.objectID}
      item={item}
      onRemoveItem={onRemoveItem}
    />
  ));

//Optimization prevent re-rendering
const List = React.memo( //here
  ({ list, onRemoveItem }) =>
    console.log('B:List') ||
    list.map(item => (
      <Item
        key={item.objectID}
        item={item}
        onRemoveItem={onRemoveItem}
      />
    ))

);

//Optimization heavy computation
const getSumComments = stories => {
  console.log('C');

  return stories.data.reduce(
    (result, value) => result + value.num_comments,
    0
  );
};


const App = () => {
  ...

  //const sumComments = getSumComments(stories); OK for a non-heavy computation
  const sumComments = React.useMemo(() => getSumComments(stories), [
    stories,
  ]);

  return (
    <div>

      <h1>My Hacker Stories with {sumComments} comments.</h1>

      ...
    </div>
  );
};

//TypeScript
npm install --save typescript @types/node @types/react
npm install --save typescript @types/react-dom @types/jest

mv src/index.js src/index.tsx
mv src/App.js src/App.tsx

//Testing Jest
// test suite
describe('truthy and falsy', () => {
  // test case
  it('true to be true', () => {
    // test assertion
    expect(true).toBe(true);
  });

  // test case
  it('false to be false', () => {
    // test assertion
    expect(false).toBe(false);
  });
});

import React from 'react';
import renderer from 'react-test-renderer';

import App, { Item, List, SearchForm, InputWithLabel } from './App';

import React from 'react';
import renderer from 'react-test-renderer';

import App, { Item, List, SearchForm, InputWithLabel } from './App';

describe('Item', () => {
  const item = {
    title: 'React',
    url: 'https://reactjs.org/',
    author: 'Jordan Walke',
    num_comments: 3,
    points: 4,
    objectID: 0,
  };

  it('renders all properties', () => {
    const component = renderer.create(<Item item={item} />);

    expect(component.root.findByType('a').props.href).toEqual(
      'https://reactjs.org/'
    );
  });
});

//mock library
import axios from 'axios';

jest.mock('axios');