export default function DoSomething() {

}

export const DoSomethingElse = () => {

}

let a = 10;
let n = a == 10 && 'Jack';

const Person = {
    name: 'Pedro',
    age: 25,
    isMarried: false,
}

const {name, age, isMarried} = Person;

const Person2 = {...Person, name: 'Jack'}; //same as Person, except name is now Jack

//JS map, filter, reduce. Which is in place, which is a copy
let names = ['A', 'B', 'C'];

names.map((name) => {
     console.log(name);
})

names.filter((name) => {
    return name != 'D'
})

