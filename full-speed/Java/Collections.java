import java.time.DayOfWeek;
import java.time.LocalDate;
import java.time.LocalTime;
import java.time.Month;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.IntSummaryStatistics;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.StampedLock;
import java.util.function.BiFunction;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.IntUnaryOperator;
import java.util.function.UnaryOperator;
import java.util.stream.Collector;
import java.util.stream.Collectors;

import javax.swing.text.html.HTMLDocument.Iterator;

public class ArrayListDemo {
    public static void main(String args[] {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(2); //add at the end
        list.add(1, 50); //add 50 at index 1
        System.out.println(list);

        List<Integer> newList = new ArrayList<>();
        newList.add(150);
        newList.add(160);

        list.addAll(newList); //add all elements in newList to list
        System.out.println(list);
        System.out.println("The size of the list is: " + list.size());

        //Iterating
        for(int i = 0; i < list.size(); i++) { //Simple for loop
            System.out.println(list.get(i));
        }

        for (Integer i : list) { //Enhanced for loop
            System.out.println(i);
        }

        //Iterator
        Iterator<Integer> itr = list.iterator();

        while(itr.hasNext) {
            int next = itr.next();
            if(next == 30) {
                itr.remove(); //Correct way to remove in loop/ not in for loop
            }
        }

        //forEachRemaining()
        Iterator<Integer> newItr = list.iterator();
        newItr.forEachRemaining(element -> System.out.println(element));

        //ListIterator
        ListIterator<Integer> listIterator = list.listIterator();

        while(listIterator.hasNext()) {
            System.out.println("Next element is " + listIterator.next() + " and next index is " + listIterator.nextIndex());
        }

        while(listIterator.hasPrevious()) {
            System.out.println("Previous element is " + listIterator.previous() + " and previous index is " + listIterator.previousIndex());
        }

        Collections.sort(list);
        Collections.sort(list, Collections.reverseOrder());

        List<Integer> sortedList = list.stream().sorted().collect(Collectors.toList());
        List<Integer> sortedReverseList = list.stream().sorted(Comparator.reverseOrder()).collect(Collectors.toList());

        list.set(1, 100);
        list.remove(1);
        list.clear();
    })
}

class Employee implements Comparable<Employee> {
    String name;
    int age;

    public Employee(String name, int age) {
        super();
        this.name = name;
        this.age = age;
    }

    @Override
    public int compareTo(Employee emp) {
        return (this.age - emp.age);
    }
}


public class LinkedListDemo {
    public static void main(Strings args[]) {
        LinkedList<Integer> linkedList = new LinkedList<>();

        linkedList.add(1);
        linkedList.add(2);
        linkedList.addLast(3);
        linkedList.addFirst(10);;
        linkedList.add(2, 20);

        List<Integer> list = new ArrayList<>();
        list.add(101);
        list.add(102);

        linkedList.addAll(3, list);
        System.out.println(linkedList);

        System.out.println(linkedList.getFirst());
        System.out.println(linkedList.getLast());
        System.out.println(linkedList.get(2));

        linkedList.remove(); //first element
        linkedList.removeLast(); //last element
        linkedList.remove(new Integer(2)); //first occurence of 2
        linkedList.removeLastOccurrence(new Integer(4)); //last occurence of 4

        Collections.sort(linkedList);
    }
}

public class CopyOnWriteArrayListDemo {
    public static void main(String args[]) {
        List<String> list = new CopyOnWriteArrayList<>();
        list.add("Apple");
        list.add("Banana");
        list.add("Orange");

        list.forEach(System.out::println);

        Iterator<String> itr = list.iterator();
        while(itr.hasNext()) {
            System.out.println(itr.next());
        }
    }
}

public class HashSetDemo {
    public static void main(String args[]) {
        Set<Integer> set = new HashSet<>();

        set.add(23);
        set.add(34);
        set.add(56);

        System.out.println(set.contains(23));
         
        for(int i : set) {
            System.out.println(i);
        }

        Iterator<Integer> itr = set.iterator();
        while(itr.hasNext()) {
            System.out.println(itr.next());
        }

        set.forEach(System.out::println);

        //sort --> convert to other Collection: List, TreeSet, or LinkedHashSet
        List<Integer> list = new ArrayList<>(set);
        Collections.sort(list);
        list.forEach(System.out::println);

        set.remove(23);
        set.clear();
        System.out.println(set.isEmpty());
    }
}

public class TreeSetDemo {
    public static void main(String args[]) {
        List<Integer> list = new LinkedList<>();
        list.add(21);
        list.add(32);
        list.add(44);
        list.add(11);
        list.add(54);

        TreeSet<Integer> set = new TreeSet<>(list);

        set.add(55);
        
        TreeSet<Integer> reverseSet = new TreeSet<>(Comparator.reverseOrder());
        reverseSet.add(21);
        reverseSet.add(32);
        reverseSet.add(44);
        reverseSet.add(11);
        reverseSet.add(54);

        System.out.println("First element in TreeSet: " + set.first());
        System.out.println("Last element in TreeSet: " + set.last());
        System.out.println("All elements less than 40 : " + set.headSet(40, false));
        System.out.println("All elements greater than 40: " + set.tailSet(40, false));

        System.out.println("Remove 44 from TreeSet: " + set.remove(new Integer(44)));
        System.out.println("TreeSet is empty: " + set.isEmpty());
        System.out.println("Size: " + set.size());
        System.out.println("Contains 44: " + set.contains(new Integer(44)));
    }
}

public class HashMapDemo {
    public static void main(String args[]) {
        Map<String, Integer> stockPrice = new HashMap<>();

        stockPrice.put("Oracle", 56);
        stockPrice.put("Fiserv", 117);
        stockPrice.put("BMW", 73);
        stockPrice.put("Microsoft", 213);

        System.out.println(stockPrice.get("Oracle"));

        System.out.println(stockPrice.get("Google")); //null
        System.out.println(stockPrice.getOrDefault("Google", 100)); //100

        stockPrice.replace("Oracle", 56, 76);
        stockPrice.replace("Fiserv", 100);
        stockPrice.replaceAll((k,v) -> v + 10);

        stockPrice.remove("Google"); //null
        stockPrice.remove("BMW", 45);

        System.out.println(stockPrice.containsKey("Oracle"));
        System.out.println(stockPrice.containsValue(73));

        Set<String> keys = stockPrice.keySet();
        for(String key : keys) {
            System.out.println(key);
        }

        Collection<Integer> values = stockPrice.values();
        for(Integer value : values) {
            System.out.println(value);
        }

        System.out.println(stockPrice.isEmpty());

        Set<Entry<String, Integer>> entrySet = stockPrice.entrySet();
        for(Entry<String, Integer> entry : entrySet) {
            System.out.println("Company name: " + entry.getKey() + " Stock Price: " + entry.getValue());
        }

        Iterator<Entry<String, Integer>> itr = entrySet.iterator();
        while(itr.hasNext()) {
            Entry<String,Integer> entry = itr.next();
            if(entry.getKey().equals("Oracle")) {
                itr.remove();
            }
        }

        stockPrice.forEach((key,value) -> System.out.println("Company Name: " + key + " Stock Price: " + value));

        Map<String, Integer> map = new HashMap<>();
        map.put("India", 5);
        map.put("USA", 3);
        map.put("China", 5);
        map.put("Russia", 6);

        map.compute("India", (k,v) -> v == null ? 10 : v + 1);
        map.compute("Vietnam", (k,v) -> v == null ? 10 : v + 1);

        map.computeIfAbsent("Vietnam", k -> k.length());

        map.computeIfPresent("India", (k,v) -> v == null ? 10 : v + 1);

        Map<String, Integer> map1 = new HashMap<>();
        map1.put("Jay", 5000);
        map1.put("Rahul", 3000);
        map1.put("Nidhi", 4500);
        map1.put("Amol", 60000);

        Map<String, Integer> map2 = new HashMap<>();
        map2.put("Jay", 7000);
        map2.put("Rahul", 4500);
        map2.put("Nidhi", 1200);
        map2.put("Saurav", 25000);

        map1.forEach((key,value) -> map2.merge(key, value, (v1, v2) -> v1 + v2));
    }
}

public class TreeMapDemo {
    public static void main(String args[]) {
        TreeMap<String, Integer> reverseMap = new TreeMap<>(Comparator.reverseOrder());
        reverseMap.put("Oracle", 43);
        reverseMap.put("Microsoft", 56);
        reverseMap.put("Apple", 43);
        reverseMap.put("Novartis", 87);

        TreeMap<String, Integer> finalMap = new TreeMap<>();
        finalMap.putAll(reverseMap);

        Entry<String, Integer> firstEntry = finalMap.firstEntry();
        System.out.println("Smallest key: " + firstEntry.getKey() + ", Value: " + firstEntry.getValue());

        Entry<String, Integer> lastEntry = finalMap.lastEntry();

        finalMap.remove("Oracle");
        finalMap.replace("Apple", 90);
        finalMap.replace("Apple", 50, 76);
    }
}

public class LinkedHashMapDemo {
    public static void main(String args[]) {
        HashMap<String, Integer> stocks = new LinkedHashMap<>(16, 0.75f, false);

        stocks.put("Apple", 123);
        stocks.put("BMW", 54);
        stocks.put("Google", 87);
        stocks.put("Microsoft", 232);
        stocks.put("Oracle", 76);
    }
}

public class ConcurrentHashMapDemo {
    public static void main(String args[]) {
        ConcurrentHashMap<String, Integer> stocks = new ConcurrentHashMap<>();

        stocks.put("Google", 123);
        stocks.put("Microsoft", 654);
        stocks.put("Apple", 345);
        stocks.put("Tesla", 999);

        stocks.putIfAbsent("Apple", 1000);
    }
}

public class EnumMapDemo {
    EnumMap<DayOfWeek, Integer> enumMap = new EnumMap<>(DayOfWeek.class);

    enumMap.put(DayOfWeek.MONDAY, 5);
    enumMap.put(DayOfWeek.WEDNESDAY, 23);

    System.out.println(enumMap.get(DayOfWeek.MONDAY));

    enumMap.remove(DayOfWeek.MONDAY);
}

public class ArrayDemo {
    public static void main(String args[]) {
        int[] numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

        int index = Arrays.binarySearch(numbers, 4);
        int index2 = Arrays.binarySearch(1, 4, 5);

        Employee[] employees = { new Employee("Jay", 123), new Employee("Roy", 124), new Employee("Nikki", 125)};

        int index3 = Arrays.binarySearch(employees, new Employee("Roy", 124), (emp1, emp2) -> emp1.age - emp2.age);

        Integer[] numbers2 = { 10, 2, 32, 12, 15, 76, 17, 48, 79, 9 };

        Arrays.sort(numbers2);
        //Arrays.parallelSort(numbers);

        int[] newArray1 = Arrays.copyOf(numbers, numbers.length);
        int[] newArray2 = Arrays.copyOf(numbers, 20);
        int[] newArray3 = Arrays.copyOfRange(numbers, 0, 5);

        Employee[] copiedArray = Arrays.copyOf(employees, 2);

        List<Integer> numbersList = Arrays.asList(numbers);

        boolean isEqual = Arrays.equals(numbers, numbers2);

        Arrays.fill(numbers2, 20);

        System.out.println("The min: " + Collections.min(numbers));
        System.out.println("The max: " + Collections.max(numbers));
        System.out.println("Frequency of 2: " + Collections.frequency(numbers, 9));
        
        List<Integer> unmodifiableList = Collections.unmodifiableList(numbers);
    }
}
class Person {
    String name;
    int age;
    String country;

    Person(String name, int age, String country) {
        this.name = name;
        this.age = age;
        this.country = country;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public String getCountry() {
        return country;
    }
}


public class ComparatorLambdaDemo {
    public static List<Person> getPersons(List<Person> persons) {
        Collections.sort(persons, (p1, p2) -> p1.getName().compareTo(p2.getName()));
        return persons;
    }
}

public class PredicateDemo {
    static boolean isPersonEligibleForMembership(Person person, Predicate<Person> predicate) {
        return predicate.test(person);
    }

    public static void main(String args[]) {
        Person person = new Person("Alex", 23);

        Predicate<Person> greaterThanEighteen = (p) -> p.age > 18;
        Predicate<Person> lessThanSixty = (p) -> p.age < 60;

        Predicate<Person> predicate = greaterThanEighteen.and(lessThanSixty);

        boolean eligible = isPersonEligibleForMembership(person, predicate);
    }
}

public class SupplierDemo {
    static boolean isPErsonEligibleForVoting(Supplier<Person> supplier, Predicate<Person> predicate) {
        return predicate.test(supplier.get());
    }
    public static void main(String args[]) {
        Supplier<Person> supplier = () -> new Person("Alex", 23);
        Predicate<Person> predicate = (p) -> p.age > 18;

        boolean eligible = isPErsonEligibleForVoting(supplier, predicate);

        IntSupplier intSupplier = () -> (int)(Math.random() * 10);
    }
}

public class ConsumerDemo {
    public static void main(String args[]) {
        BiConsumer<String, String> greet = (s1, s2) -> System.out.println(s1 + s2);
        greet.accept("Hello", "World");
    }
}

public class FunctionInterfaceDemo {
    public static void main(String args[]) {
        Function<String, Integer> lengthFunction = str -> str.length();

        System.out.println("String lenght: " + lengthFunction.apply("This is awesome!!!"));

        BiFunction<Integer, Integer, Integer> add = (a, b) -> a + b;

        System.out.println("Sum = " + add.apply(2, 3));
    }
}

public class UnaryOperatorTest {
    public static void main(String args[]) {
        Person person = new Person();
        UnaryOperator<Person> operator = (p) -> {
            p.name = "John";
            p.age = 34;
            return p;
        };
        operator.apply(person);

        IntUnaryOperator intOperator = num -> num * num;
        System.out.println(intOperator.applyAsInt(25));
    }
}

public class BinaryOperatorDemo {
    public static void main(String args[]) {
        Person person1 = new Person("Alex", 23);
        Person person2 = new Person("Daniel", 56);
        BinaryOperator<Person> operator = (p1,p2) -> {
            p1.name = p2.name;
            p1.age = p2.age;
            return p1;
        }

        operator.apply(person1, person2);
    }
}

public class CapturingLambdaDemo {
    public static void main(String args[]) {
        Function<Integer, Integer> multiplier = getMultiplier();
        System.out.println(multiplier.apply(10));
    }

    public static Function<Integer, Integer> getMultiplier() {
        int i = 5;
        // The below lambda has copied the value of i
        Function<Integer,Integer> multiplier = t -> t * i;
        // If you change the value of i here, the lambda will have old value
        // This is not allowed and code will not compile
        // i = 7;
        return multiplier;
    }
}

public class StreamDemo {
    public static void main(String args[]) {
        List<Integer> list = new ArrayList<>();
        list.add(1);
        list.add(12);
        list.add(23);
        list.add(45);
        list.add(6);

        Stream<String> stream = list.stream();
        stream.forEach(p -> System.out.println(p));

        List<Person> listP = new ArrayList<>();
        listP.add(new Person("Dave", 23));
        listP.add(new Person("Joe", 18));
        listP.add(new Person("Ryan", 54));
        listP.add(new Person("Iyan", 5));
        listP.add(new Person("Ray", 63));

        listP.stream()
            .filter(person -> person.getName() != null)
            .filter(person -> person.getAge() > 18)
            .filter(person -> person.getAge() < 60)
            .forEach(System.out::println);
    }
}

public class StreamMapDemo {
    public static void main(String args[]) {
        List<String> list = new ArrayList<>();
        list.add("Dave");
        list.add("Joe");
        list.add("Ryan");
        list.add("Iyan");
        list.add("Ray");

        list.stream()
                .mapToInt(name -> name.length())
                .forEach(System.out::println);

        List<List<String>> list1 = new Array<>();
        list1.add(Arrays.asList("a","b","c"));
        list1.add(Arrays.asList("d","e","f"));
        list1.add(Arrays.asList("g","h","i"));
        list1.add(Arrays.asList("j","k","l"));

        Stream<List<String>> stream1 = list1.stream;
        // doesn't work on stream of collections
        Stream<List<String>> stream2 = stream1.filter(x -> "a".equals(x.toString()));

         // Flattened the stream.
        Stream<String> stream3 = stream1.flatMap(s -> s.stream());
        //Applied filter on flattened stream.
        Stream<String> stream4 = stream2.filter(x -> "a".equals(x));
        
        stream3.forEach(System.out::println);
    }
}

public class MethodReferenceDemo {
    public static int getLength(String str) {
        return str.length();
    }

    public static void main(String args[]) {
        List<String> list = new ArrayList<>();
        list.add("done");
        list.add("word");
        list.add("pracetice");
        list.add("fake");

        list.stream()
            .mapToInt(str -> MethodReferenceDemo.getLength(str))
            .forEach(System.out::println);
        
        list.stream()
            .mapToInt(MethodReferenceDemo::getLength)
            .forEach(System.out::println);

    }
}
class Employee2 {
    String name;
    int age;
    int salary;
    String country;

    Employee2(String name, int age, int salary, String country) {
        this.name = name;
        this.age = age;
        this.salary = salary;
        this.country = country;
    }

    public String getName() {
        return name;
    }

    public int getAge() {
        return age;
    }

    public Integer getSalary() {
        return salary;
    }

    public String getCountry() {
        return country;
    }

    @Override
    public String toString() {
        return "Employee{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", salary=" + salary +
                '}';
    }
}

public class OptionalDemo {
    Map<Integer, Employee> empMap = new HashMap<>();
    public Optional<Employee> getEmployee(Integer employeeId) {
        return Optional.ofNullable(empMap.get(employeeId));
    }  

    public static void main(String args[]) {
        OptionalDemo demo = new OptionalDemo();
        Optional<Employee> emp = demo.getEmployee(123);
        if(emp.isPresent()) {
            System.out.println(emp.get().getName());
        } else {
            System.out.println("No employee returned");
        }
    }
}

public class SliceDemo {
    public static main(String args[]) {
        List<String> countries = new ArrayList<>();
        countries.add("India");
        countries.add("USA");
        countries.add("China");
        countries.add("India");
        countries.add("UK");
        countries.add("China");

        countries.stream()
                .skip(2)
                .distinct()
                .limit(3)
                .forEach(System.out::println);
    }
}

public class MatchDemo {
    public static void main(String args[]) {
        List<Person> list = new ArrayList<>();
        list.add(new Person("Dave", 23, "India"));
        list.add(new Person("Joe", 18,"USA"));
        list.add(new Person("Ryan", 54,"Canada"));
        list.add(new Person("Iyan", 5,"India"));
        list.add(new Person("Ray", 63,"China"));

        boolean anyCanadian = list.stream()
                                    .anyMatch(p -> p.getCountry().equals("Canada"));
        boolean allCanadian = list.stream().allMatch(p -> p.getClass().equals("Canada"));

        boolean noneRussian = list.stream().noneMatch(p -> p.getCountry().equals("Russia"));

        Optional<Person> person1 = list.stream().filter(p -> p.getCountry().equals("India")).findFirst();

        Optional<Person> person2 = list.stream().filter(p -> p.getClass().equals("Russia")).findAny();
    }
}

public class ReductionDemo {
    public static void main(String args[]) {
        List<Employee2> list = new ArrayList<>();
        list.add(new Employee2("Dave", 23, 20000));
        list.add(new Employee2("Joe", 18,40000));
        list.add(new Employee2("Ryan", 54,100000));
        list.add(new Employee2("Iyan", 5,34000));
        list.add(new Employee2("Ray", 63,54000));

        Optional<Integer> totalSalary = list.stream().map(p -> p.getSalary()).reduce((p, q) -> p + q);

        if(totalSalary.isPresent()) {
            System.out.println("The total salary is " + totalSalary.get());
        }

        int totalSalary2 = list.stream().mapToInt(p -> p.getSalary()).sum();

        List<Integer> listN = new ArrayList<>();
        listN.add(1);
        listN.add(2);
        listN.add(3);
        listN.add(4);
        listN.add(5);

        int totalSum = listN.stream().reduce(5, (partialSum, num) -> partialSum + num);

        int totalSum2 = listN.parallelStream().reduce(0, (partialSum, num) -> partialSum + num, Integer::sum);

        Optional<Integer> max = listN.stream().max(Comparator.naturalOrder());

        Optional<Integer> min = listN.stream().min(Comparator.naturalOrder());
    }
}


public class CollectorsDemo {
    public static void main(String args[]) {
        List<Employee2> employeeList = new ArrayList<>();
        employeeList.add(new Employee2("Alex", 23, 23000, "USA"));
         employeeList.add(new Employee2("Ben" , 63, 25000, "India"));
        employeeList.add(new Employee2("Dave" , 34, 56000, "Bhutan"));
        employeeList.add(new Employee2("Jodi" , 43, 67000, "China"));
        employeeList.add(new Employee2("Ryan" , 53, 54000, "Libya"));

        List<String> empName = employeeList.stream()
        .map(emp -> emp.getName())
        .collect(Collectors.toList());

        Set<String> empNameSet = employeeList.stream()
        .map(emp -> emp.getName())
        .collect(Collectors.toSet());

        LinkedList<String> empNameLL = employeeList.stream()
        .map(emp -> emp.getName())
        .collect(Collectors.toCollection(LinkedList::new));

        List<String> list = new ArrayList<>();
        list.add("done");
        list.add("far");
        list.add("away");
        list.add("again");

        Map<String, Integer> nameMap = list.stream()
        .collect(Collectors.toMap(s -> s, s -> s.length(), (s1, s2) -> s1, HashMap::new));

        List<String> unmodifiableList = list.stream()
        .collect(Collectors.collectingAndThen(Collector.toList(), Collections::unmodifiableList));

        long count = employeeList.stream()
            .filter(emp -> emp.getAge() > 30)
            .collect(Collectors.counting());

        int count = employeeList.stream()
            .collect(Collectors.summingInt(emp -> emp.getSalary()));

        double average = employeeList.stream()
            .collect(Collectors.averagingInt(emp -> emp.getSalary()));

        Optional<Employee2> employee = employeeList.stream()
        .collect(Collectors.minBy(Comparator.comparing(Employee2::getSalary)));

        Optional<Employee2> employeeMax = employeeList.stream()
        .collect(Collectors.maxBy(Comparator.comparing(Employee2::getSalary)));

        IntSummaryStatistics summarizingInt = Stream.of("1", "2", "3")
        .collect(Collectors.summarizingInt(Integer::parseInt));

        String joinedString = Stream.of("hello", "how", "are", "you")
        .collect(Collectors.joining(" ", "prefix", "suffix"));

        Map<String,List<Employee2>> empMap1 = employeeList.stream()
        .collect(Collectors.groupingBy(Employee2::getCountry));
        
        Map<String, Optional<Employee2>> empMap2 = employeeList.stream()
        .collect(Collectors.groupingBy(Employee2::getCountry, Collectors.maxBy(Comparator.comparingInt(Employee2::getSalary))));

        Map<String, Set<Employee2>> empMap3 = employeeList.stream()
        .collect(Collectors.groupingBy(Employee2::getCountry, HashMap::new, Collectors.toSet()));

        ConcurrentMap<String, List<Employee2>> empMap4 = employeeList.parallelStream()
        .collect(Collectors.groupingBy(Employee2::getCountry));

        Map<Boolean, List<Employee2>> empMap5 = employeeList.stream()
        .collect(Collectors.partitioningBy(emp -> emp.getAge() > 30));
    }
}

public class ParallelStreamDemo {
    public static void main(String args[]) {
        Stream.of(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            .parallel()
            .forEach(num -> System.out.println(num + " " + Thread.currentThread().getName()));
    }
}

public class LazyEvaluationDemo {
    public static void main(String args[]) {
        List<Integer> data = Array.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        Optional<Integer> number = data.stream()
            .filter(num -> {
                System.out.println("Processing first filter for number " + num);
                return num > 5;
            })
            .filter(num -> {
                System.out.println("Processing second filter for number " + num);
                return num % 3 == 0;
            })
            .findFirst();

        System.out.println(number.get());
    }
}

public class CompletableFutureDemo {
    public static void main(String args[]) {
        CompletableFuture<String> completedFutute = CompletableFuture.completedFuture("Hello World");

        try {
            System.out.println(completedFutute.get());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
            try {
                TimeUnit.SECONDS.sleep(5);
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            System.out.println("This will print immediately " + Thread.currentThread().getName());
        });

        System.out.print("This will print immedidately " + Thread.currentThread().getName());

        try {
            future.get();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }

        System.out.println("This will print after 5 seconds " + Thread.currentThread().getName());
    }

    public CompletableFuture<Integer> getSquareAsynchronously(int num) throws InterruptedException {
        CompletableFuture<Integer> completableFuture = new CompletableFuture<>();

        Executors.newCachedThreadPool().submit(() -> {
            Thread.sleep(500);

            completableFuture.complete(num * num);
            return null;
        });

        return completableFuture;
    }

    public void supplyA() {
        CompletableFuture<String> future = CompletableFuture.supplyAsync(() -> {
            try {
                TimeUnit.SECONDS.sleep(5);
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            return "Hello World";
        });

        System.out.println("THis will print immediately");

        try {
            System.out.println(future.get());
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }

        System.out.println("This will print after 5 seconds");
    }

    public void thenApply() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
            try {
                TimeUnit.SECONDS.sleep(1);
                System.out.println(Thread.currentThread().getName());
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            return 50;
        });

        CompletableFuture<Integer> resultFuture = future.thenApply(num -> {
            System.out.println(Thread.currentThread().getName());
            return num * 2;
        });

        future.thenAccept(num -> {
            System.out.println(Thread.currentThread().getName());
            System.out.println("The value is " + num);
        });

        future.thenRun(() -> {
            System.out.println(Thread.currentThread().getName());
            System.out.println("Hello");
        });

        try {
            System.out.println(resultFuture.get());
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }
    }

    public void thenApplySync() {
        CompletableFuture<Integer> resultFuture = future.thenApplySync(num -> {
            System.out.println(Thread.currentThread().getName());
            return num * 2;
        });

        ExecutorService executor = Executors.newFixedThreadPool(5);

        CompletableFuture<Integer> resultFuture = future.thenApplySync(num -> {
            System.out.println(Thread.currentThread().getName());
            return num * 2;
        }, executor);
    }

    public void Chaining() {
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(() -> {
            try {
                TimeUnit.SECONDS.sleep(1);
                System.out.println(Thread.currentThread().getName());
            } catch (InterruptedException e) {
                throw new IllegalStateException(e);
            }
            return 50;
        });

        CompletableFuture<Integer> resultFuture = future.thenCompose(num -> CompletableFuture.supplyAsync(() -> num * 2));

        CompletableFuture<Integer> resultCombine = future.thenCombine(CompletableFuture.supplyAsync(() -> 20), (num1, num2) -> num1 + num2);
    }

    public void Combining() {
        CompletableFuture<Integer> future1 = CompletableFuture.supplyAsync(() -> 50);
        CompletableFuture<Integer> future2 = CompletableFuture.supplyAsync(() -> 40);
        CompletableFuture<Integer> future3 = CompletableFuture.supplyAsync(() -> 30);

        CompletableFuture<Void> finalFuture = CompletableFuture.allOf(future1, future2, future3);

        try {
            finalFuture.get();
        } catch (Exception e) {
            e.printStackTrace();
        }

        Optional<Integer> maxElement = Stream.of(future1, future2, future3)
        .map(CompletableFuture::join)
        .max(Integer::compareTo);

        CompletableFuture<Object> firstCompletedFuture = CompletableFuture.anyOf(future1, futu2, futur2);
    }
}

public class StampedLockDemo1 {

    static Map<String, Integer> data = new HashMap<>();
    static StampedLock lock = new StampedLock();

    public static Integer readDataFromMap(String key) {
        long stamp = lock.tryReadLock();
        int result = 0;
        if(stamp != 0L) {
            try {
                result = data.get(key);
            } finally {
                lock.unlockRead(stamp);
            }
        }
        return result;
    }

    public static void writeDataToMap(String key, Integer value) {
        long stamp = lock.tryWriteLock();
        if (stamp != 0L) {
            try {
                data.put(key, value);
            } finally {
                lock.unlockWrite(stamp);
            }
        }
    }

    public static void main(String args[]) {
        writeDataToMap("ray", 123);
        Integer val = readDataFromMap("ray");
        System.out.println(val);
    }
}

public class DateTimeDemo {
    public static void main(String args[]) {
        LocalDate date = LocalDate.now();

        LocalDate date1 = LocalDate.of(2019, 05, 03);

        LocalDate date2 = LocalDate.of(2019, Month.AUGUST, 03);

        LocalDate date3 = LocalDate.parse("2015-02-12");

        LocalDate date4 = LocalDate.parse("12/02/2012", DateTimeFormatter.ofPattern("MM/dd/yyyy"));

        LocalDate date5 = LocalDate.parse("2015-02-12").plusDays(4);

        DayOfWeek dayOfWeek = LocalDate.parse("2017-04-06").getDayOfWeek();

        boolean isBefore = LocalDate.parse("2020-03-12")
        .isBefore(LocalDate.parse("2018-06-14"));
    
        LocalTime time = LocalTime.now();

        LocalTime time2 = LocalTime.of(11, 25, 03);

        LocalTime time3 = LocalTime.parse("08:27");

        LocalTime time4 = LocalTime.parse("08:27", DateTimeFormatter.ofPattern("HH:mm"));

        time = LocalTime.parse("12:54:53").plus(4, ChronoUnit.MINUTES);

        int minute = LocalTime.parse("07:45").getMinute();

        boolean isBeforeTime = LocalTime.parse("06:23").isBefore(LocalTime.parse("07:50"));
    }
}
