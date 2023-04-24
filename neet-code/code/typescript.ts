//Types
let score: number;

function add(a: number, b: number): number {
  return a + b
}

const multiply = (a: number, b: number): number => a * b;

//optional
function add(a:number, b?:number):number {
  return a + (b || 0); //b may be undefined
}

const firstName = "Bob";
typeof firstName; //'Bob' string literal type

//any
const formValues: { [field: string]: any } = {
  firstName: "Bob",
  surname: "Smith",
  age: 30
};

//unknown type
function add(a: unknown, b: unknown) {
  if (typeof a === "number" && typeof b === "number") {
    return a + b;
  }
  return 0;
}

function isPerson(person: any): person is Person {
  return "id" in person && "name" in person;
}

//void
function logMessage(message: string): void {
  console.log(message);
}

//type assertions
function getAge(id: number): any {
  return 42;
}
function calcDiscount(age: number) {
  return age / 100;
}

const discount3 = calcDiscount(getAge(1) as number);

//typed array
const numbers: Array<number> = [];
const numbers: string[] = ["one", "two", "three"];
function logScores(firstName: string, ...scores: number[]) {
  console.log(firstName, scores);
}

//tuple
const tomScore: [string, number] = ["Tom", 70];
const benScores: [string, ...number[]] = ["Ben", 50, 75, 85]

//object
const tomScore: { name: string; score: number; } = {
  name: "Tom",
  score: 70
}

//type alias
type Score = { name: string; score: number; };
const bobScore: Score = { name: "Bob", score: 80 };

//interface
interface ButtonProps {
  readonly text: string;
  onClick?: () => void; //optional
}

interface ColoredButtonProps extends ButtonProps {
  color: string;
}

const GreenBuyButton: ColoredButtonProps = {
  color: "Green",
  text: "Buy",
  onClick: () => console.log("Buy")
}

//union type
let age: number | null | undefined;
age = null;      // okay
age = 30;        // okay
age = "30";      // error

//intersection type 
type Name = {
  firstName: string;
  lastName: string;
}
type PhoneNumber = {
  landline: string;
  mobile: string;
}

type Contact = Name & PhoneNumber;

const fred: Contact =  {
  firstName: "Fred",
  lastName: "Smith",
  landline: "0116 4238978",
  mobile: "079543 4355435"
}

console.log(fred);

//Generic types
let scores: Array<number>;
scores = [70, 65, 75];

const response: Promise<Response> = fetch("https://swapi.dev/api/");
response.then(res => console.log(res.ok));

type Action = {
  type: "fetchedName";
  data: string;
}
type ImmutableAction = Readonly<Action>;

type Contact = {
  name: "Bob";
  email: "bob@someemail.com";
}

//Partial<Contact> would be equivalent to
type Contact = {
  name?: "Bob";
  email?: "bob@someemail.com";
}

type Result = {
  firstName: string;
  surname: string;
  score: number;
}
type ResultRecord = Record<string, Result>;

//generic function 
function findFirst<ItemType>(
  array: ItemType[],
  match: ItemType
): ItemType | null {
  return array.indexOf(match) === -1
    ? null
    : array.filter((item) => item === match)[0];
}

const firstOrNull = <ItemType>(
  array: ItemType[]
): ItemType | null =>
  array.length === 0 ? null : array[0];

//generic interface
interface Form<T> {
  values: T;
}

interface Contact {
  name: string;
  email: string;
}

const contactForm: Form<Contact> = {
  values: {
    name: "Bob",
    email: "bob@someemail.com"
  }
}

console.log(contactForm);

//generic alias
type Form<T> = {
  errors: {
    [P in keyof T]?: string;
  };
  values: T;
};

//generic class
class List<ItemType> {
  private items: ItemType[] = [];  
  
  add(item: ItemType) {
    this.items.push(item);
  }
}

const numberList = new List<number>();
numberList.add(1);

//React props
//strongly-typed
const Hello = (props: { who: string }) => (
  <p>Hello, {props.who}</p>
);

const Hello = ({ who }: { who: string }) => ( //destructuring
  <p>Hello, {who}</p>
);

type Props = { who: string }
function Hello({ who }: Props) {
  return <p>Hello, {who}</p>;
}

//React.FC type
const Hello: React.FC<Props> = ({ who }) => (
  <p>Hello, {who}</p>
);

//props optional
type Props = { 
  who: string; 
  message?: string 
};
const Hello = ({ who, message }: Props) => (
  <React.Fragment>
    <p>Hello, {who}</p>
    {message && <p>{message}</p>}
  </React.Fragment>
);

//default props
type Props = { who: string; message?: string };
const Hello = ({ who, message }: Props) => (
  <React.Fragment>
    <p>Hello, {who}</p>
    {message && <p>{message}</p>}
  </React.Fragment>
);
Hello.defaultProps = {
  message: "How are you?"
};

//object prop
type Address = {
  line1: string; 
  line2: string; 
  state: string; 
  zipcode: string;
}
type Who = {
  name: string;
  friend: boolean;
  address?: Address;
}
type Props = {
  who: Who;
  message?: string;
}
const Hello = ({ who, message = "How are you?" }: Props) => (
  <React.Fragment>
    <p>
      Hello, {who.name}
      {who.friend && " my friend"}
    </p>
    {message && <p>{message}</p>}
  </React.Fragment>
);

//function props
type Props = {
  who: Who;
  message?: string;
  renderMessage?: (message: string) => React.ReactNode;
}

const Hello = ({
  who,
  renderMessage,
  message = "How are you?"
}: Props) => (
  <React.Fragment>
    <p>
      Hello, {who.name}
      {who.friend && " my friend"}
    </p>
    {message && (renderMessage ? renderMessage(message) : <p>{message}</p>)}
  </React.Fragment>
);

<Hello
  who={{ name: "Bob", friend: true }}
  message="Hey, how are you?"
  renderMessage={m => <i>{m}</i>}
/>

//useState
const [count, setCount] = React.useState<number | null>(null);

//useReducer
type Increment = {
  readonly type: 'increment';
  readonly incrementStep: number;
};
type Decrement = {
  readonly type: 'decrement';
  readonly decrementStep: number;
};

type Actions = Increment | Decrement;

const reducer = (state: State, action: Actions): State => {
  switch (action.type) {
    case 'increment':
      return { count: state.count + action.incrementStep };
    case 'decrement':
      return { count: state.count - action.decrementStep };
    default:
      neverReached(action);
  }
  return state;
};

const neverReached = (never: never) => {};

const Counter = ( ... ) => {
  const [state, dispatch] = React.useReducer<React.Reducer<State, Actions>>(
    reducer, 
    { count: initialCount }
  );


  return (
    <div>
      <div>{state.count}</div>
      <button onClick={() => dispatch({ type: "increment", incrementStep })}>
        Add {incrementStep}
      </button>
      <button onClick={() => dispatch({ type: "decrement", decrementStep })}>
        Subtract {decrementStep}
      </button>
    </div>
  );
};

const rootElement = document.getElementById("root");
render(
  <Counter incrementStep={1} decrementStep={2} initialCount={5} />,
  rootElement
);

//typed class
type Who = {
  name: string;
  friend: boolean;
}
type Props = {
  who: Who;
  message?: string;
  renderMessage?: (message: string) => React.ReactNode;
}

class Hello extends React.Component<Props> {
  static defaultProps = {
    message: "How are you?"
  };
  render() {
    const { who, message, renderMessage } = this.props;
    return (
      <React.Fragment>
        <p>{`Hello, ＄{who.name} ＄{who.friend && " my friend"}`}</p>
        {message && (renderMessage ? renderMessage(message) : <p>{message}</p>)}
      </React.Fragment>
    );
  }
}

//typed state
type Props = {
  initialCount?: number;
}

type State = {
  count: number;
};
class Counter extends React.Component<Props, State> {
  class Counter extends React.Component<Props, State> {
    constructor(props: {}) {
      super(props);
      this.state = {
        count: this.props.initialCount || 0
      };
    }

    private clicked: number = 0;
    private handleClick() {
      this.setState(state => ({ count: state.count + 1 }));
      this.clicked++;
    }

    render() {
      return (
        <button onClick={() => this.setState((state) => ({count: state.count + 1}))}>
          {this.state.count ? this.state.count : "Click to start counter"}
        </button>
      );
    }
  }
}

const rootElement = document.getElementById("root");
render(<Counter initialCount={5} />, rootElement);

//event handler
//inline
<input
  type="text"
  value={criteria}
  onChange={e => setCriteria(e.currentTarget.value)}
/>

const handleChange = (e: React.ChangeEvent<HTMLInputElement>) =>
  setCriteria(e.currentTarget.value);

//
type Props = {
  onSearch?: (criteria: string) => void;
};
const Searchbox = ({ onSearch }: Props) => {
  const [criteria, setCriteria] = React.useState("");

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setCriteria(e.currentTarget.value);
    if (onSearch) {
      onSearch(e.currentTarget.value);
    }
  };

  return <input type="text" value={criteria} onChange={handleChange} />;
};

//context
import * as React from "react";
import { render } from "react-dom";

const defaultTheme = "white";

const ThemeContext = React.createContext<ThemeContextType | undefined>(
  undefined
);

type Props = {
  children: React.ReactNode;
};
export const ThemeProvider = ({ children }: Props) => {
  const [theme, setTheme] = React.useState(defaultTheme);

  React.useEffect(() => {
    // We'd get the theme from a web API / local storage in a real app
    // We've hardcoded the theme in our example
    const currentTheme = "lightblue";
    setTheme(currentTheme);
  }, []);

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

const useTheme = () => React.useContext(ThemeContext);

const App = () => (
  <ThemeProvider>
    <Header />
  </ThemeProvider>
);

class Header extends React.Component {
  render() {
    return (
      <ThemeContext.Consumer>
        {value => (
          <div style={{ backgroundColor: value!.theme }}>
            <select
              value={value!.theme}
              onChange={e => value!.setTheme(e.currentTarget.value)}
            >
              <option value="white">White</option>
              <option value="lightblue">Blue</option>
              <option value="lightgreen">Green</option>
            </select>
            <span>Hello!</span>
          </div>
        )}
      </ThemeContext.Consumer>
    );
  }
}

const rootElement = document.getElementById("root");
render(<App />, rootElement);

//ref in function components
const Search: React.FC = () => {
  const input = React.useRef<HTMLInputElement>(null);
  React.useEffect(() => {
    if (input.current) {
      input.current.focus();
    }
  }, []);
  return (
    <form>
      <input ref={input} type="type" />
    </form>
  );
};

//ref in class component
class Search extends React.Component {
  private input = React.createRef<HTMLInputElement>();

  componentDidMount() {
    if (this.input.current) {
      this.input.current.focus();
    }
  }

  render() {
    return (
      <form>
        <input ref={this.input} type="type" />
      </form>
    );
  }
}

//Exercise list component
interface Props {
  items: string[];
}
export const List = ({ items }: Props) => {
  const input = React.useRef<HTMLInputElement>(null);

  const [search, setSearch] = React.useState("");
  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearch(e.currentTarget.value);
  };

  React.useEffect(() => {
    if (input.current) {
      input.current.focus();
    }
  }, []);

  return (
    <React.Fragment>
      <input
        ref={input}
        type="search"
        value={search}
        onChange={handleSearchChange}
      />
      <div>
        {items.map(item => (
          <div
            key={item}
            style={{ fontWeight: item === search ? "bold" : undefined }}
          >
            {item}
          </div>
        ))}
      </div>
    </React.Fragment>
  );
};

// *** REDUX
//action
{
  type: “WITHDRAW_MONEY”,
  amount: “$10,000”
}

//create store
import React, { Component } from "react";
import HelloWorld from "./HelloWorld";
import { createStore } from "redux";

const initialState = { tech: "React " };
const store = createStore(reducer, initialState);

class App extends Component {
  render() {
    return <HelloWorld tech={store.getState().tech}/>
  }
}


//action
{
  type: "withdraw_money",
  payload: {
     amount: "$4000"
  }
}

//reducer
function reducer (state, action) {
  switch (action.type) {
    case "is_open":
      return;  //return new state
    case "is_clicked":
      return; //return new state
    default:
      return state;
  }
}

//action creator
const setTechnology = text => ({ type: "SET_TECHNOLOGY", text });

//return state
return {
  ...state, //a copy, do not mutate original
  tech: action.tech
};

//subscribe render index.js
const render = () => ReactDOM.render(<App />, document.getElementById("root"));

render();
store.subscribe(render);






