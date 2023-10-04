//Setting up Threads
//Anonymous function

import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;
import java.util.concurrent.Callable;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.logging.Level;
import java.util.logging.Logger;

class Demonstration {
    public static void main(String args[]) throws Exception {
        Thread t = new Thread(new Runnable() {
            public void run() {
                System.out.println("Say Hello");
            }
        });
        t.start();

        ExecuteMe executeMe = new ExecuteMe();
        Thread t2 = new Thread(executeMe);
        //t2.setDaemon(true);
        t2.start();
        t2.join();
        System.out.println("Main thread exiting");
    }
}

//Explicit named function
class ExecuteMe implements Runnable {
    public void run() {
        System.out.println("Say Hello. This thread is going to sleep.");
        try {
            Thread.sleep(500);
        } catch (InterruptedException ie) {

        }
    }
}

class HelloWorld {
    public static void main(String args[]) throws InterruptedException {
        ExecuteMe executeMe = new ExecuteMe();
        Thread innerThread = new Thread(executeMe);
        innerThread.start();

        //Interrupt innerThread after waiting for 5 seconds
        System.out.println("Main thread is sleeping at " + System.currentTimeMillis() / 1000);
        Thread.sleep(5000);
        innerThread.interrupt();
        System.out.println("Main thread exiting at " + System.currentTimeMillis() / 1000);
    }

    static class ExecuteMe implements Runnable {
        public void run() {
            try {
                System.out.println("innerThread goes to sleep at " + System.currentTimeMillis() / 1000);
                Thread.sleep(1000 * 1000);
            } catch (InterruptedException ie) {
                System.out.println("innerThread interrupted at " + System.currentTimeMillis() / 1000);
            }
        }
    }


//Executor Framework
//Thread Pool
    void receiveAndExecuteClientOrdersBest() {
        int expectedConcurrentOrders = 100;
        Executor executor = Executors.newFixedThreadPool(expectedConcurrentOrders);

        while (true) {
            final Order order = waitForNextOrder();

            executor.execute(new Runnable() {
                public void run() {
                    order.execute();
                }
            });
        }
    }
}

//Timer vs Scheduled Thread Pool
class TimerDemonstration {
    public static void main( String args[] ) throws Exception {
        Timer timer = new Timer();
        TimerTask badTask  = new TimerTask() {
            @Override
            public void run() {
                //run forever
                while (true)
                ;
            }
        };

        TimerTask goodTask = new TimerTask() {
            @Override
            public void run() {
                System.out.println("Hello I'm a well-behaved task");
            }
        };

        timer.schedule(badTask, 100);
        timer.schedule(goodTask, 500);

        //By three seconds, both tasks are exptected to have launched
        Thread.sleep(3000);
    }
}

//Callable Interface
class SumTask implements Callable<Integer> {
    int n;

    public SumTask(int n) {
        this.n = n;
    }

    public Integer call() throws Exception {
        if (n <= 0)
            return 0;
        
        int sum = 0;
        for (int i = 1; i <= n; i++) {
            sum += 1;
        }

        return sum;
    }
}

class FutureDemonstation {
    static ExecutorService threadPool = Executors.newSingleThreadExecutor();

    public static void main (String args[]) throws Exception {
        System.out.println(pollingStatusAndCancelTask(10));
        threadPool.shutdown();
    }

    static int pollingStatusAndCancelTask(final int n) throws Exception {
        int result = -1;

        Callable<Integer> sumTask1 = new Callable<Integer>() {
            public Integer call() throws Exception {
                //wait for 10 milliseconds
                Thread.sleep(10);

                int sum = 0;
                for (int i = 1; i <= n; i++) {
                    sum += 1;
                }
                return sum;
            }
        };

        Callable<Void> randomTask = new Callable<Void>() {
            public Void call() throws Exception {
                //go to sleep for an hour
                Thread.sleep(3600 * 1000);
                return null;
            }
        };

        Future<Integer> f1 = threadPool.submit(sumTask1);
        Future<Void> f2 = threadPool.submit(randomTask);

        //poll for completion of first task
        try {
            //Before we poll for completion of second task, cancel the second one
            f2.cancel(true);

            //polling the future to check the status of the first submitted task
            while (!f1.isDone()) {
                System.out.println("Waiting for first task to complete");
            }
            result = f1.get();
        } catch (ExecutionException ee) {
            System.out.println("Something went wrong");
        }

        System.out.println("\nIs second taks cancelled: " + f2.isCancelled());

        return result;
    }
}

class CompletionServiceDemonstration {
    static Random random = new Random(System.currentTimeMillis());

    public static void main(String args[]) throws Exception {
        completionServiceExample();
    }

    static void completionServiceExample() throws Exception {
        class TrivialTask implements Runnable {
            int n;

            public TrivialTask(int n) {
                this.n = n;
            }

            public void run() {
                try {
                    //sleep for one second
                    Thread.sleep(random.nextInt(101));
                    System.out.println(n*n);
                } catch (InterruptedException ie) {
                    //swallow exception
                }
            }
        }

        ExecutorService threadPool  = Executors.newFixedThreadPool(3);
        ExecutorCompletionService<Integer> service = new ExecutorCompletionService<>(threadPool);

        //submit 10 trivial tasks
        for (int i = 0; i < 10; i++) {
            service.submit(new TrivialTask(i), new Integer(i));
        }

        //wait for all tasks to get donw
        int count = 10;
        while (count != 0) {
            Future<Integer> f = service.poll();
            if (f != null) {
                System.out.println("Thread" + f.get() +" got done.");
                count--;
            }
        }

        threadPool.shutdown();
    }
}

//ThreadLocal
class ThreadLocalDemonstration {
    UnsafeCounter usc = new UnsafeCounter();
    Thread[] tasks = new Thread(100);

    for (int i = 0; i < 100; i++) {
        Thread t = new Thread(() -> {
            for (int j = 0; j < 100; j++) 
                usc.increment();
                System.out.println(usc.counter.get());
        });
        tasks[i] = t;
        t.start();
    }

    for (int a = 0; a < 100; i++) {
        tasks[a].join();
    }

    System.out.println(usc.counter.get());
}

class UnsafeCounter {
    ThreadLocal<Integer> counter = ThreadLocal.withInitial(() -> 0);

    void increment() {
        counter.set(counter.get() + 1);
    }
}

//The worker thread that has to complete its tasks first

public class Worker extends Thread {
    private CountDownLatch countDownLatch;

    public Worker(CountDownLatch countDownLatch, String name) {
        super(name);
        this.countDownLatch = countDownLatch;
    }

    @Override
    public void run() {
        System.out.println("Worker " + Thread.currentThread().getName() + " started");
        try {
            Thread.sleep(3000);
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        }
        System.out.println("Worker " + Thread.currentThread().getName() + " finished");

        //each thread call countDown() method on taks completion
        countDownLatch.countDown();
    }
}

//The master thread that has to wait for the worker to complete its operation first
public class Master extends Thread {
    public Master(String name) {
        super(name);
    }

    @Override
    public void run() {
        System.out.println("Master executed " + Thread.currentThread().getName());
        try {
            Thread.sleep(2000);
        } catch (InterruptedException ie) {
            ie.printStackTrace();
        }
    }
}

public class MainClass {
    public static void main(String[] args) throws InterruptedException {
        //create countDownLatch for 2 thread
        CountDownLatch countDownLatch = new CountDownLatch(2);

        //create and start 2 threads
        Worker A = new Worker(countDownLatch, "A");
        Worker B = new Worker(countDownLatch, "B");

        A.start();
        B.start();

        //when 2 thread complete their tasks, they are returned
        countDownLatch.await();

        //Now execution of master thread has started
        Master D = new Master("Master executed");
        D.start();
    }
}

/* CyclicBarrier */
//Runnable task for each thread
class TaskCyclic implements Runnable {
    private CyclicBarrier barrier;

    public TaskCyclic(CyclicBarrier barrier) {
        this.barrier = barrier;
    }

    //await is invoked to wait for other threads
    @Override
    public void run() {
        try {
            System.out.println(Thread.currentThread().getName() + " is waiting on barrier");
            barrier.await();
            //printing after crossing the barrier
            System.out.println(Thread.currentThread().getName() + " has crossed the barrier");
        } catch (InterruptedException ex) {
            Logger.getLogger(TaskCyclic.class.getName()).log(Level.SEVERE, null, ex);
        } catch (BrokenBarrierException ex) {
            Logger.getLogger(TaskCyclic.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
}

public class MainCyclic {
    public static void main(String args[]) {
        //create CyclicBarrier with 3 parties/ threads needs to call await
        final CyclicBarrier cb = new CyclicBarrier(3, new Runnable() {
            //action that executes after the last thread arrives
            @Override
            public void run() {
                System.out.println("All partied have arrived at the barrier, lets continue execution");
            }
        });

        Thread t1 = new Thread(new TaskCyclic(cb), "Thread 1");
        Thread t2 = new Thread(new TaskCyclic(cb), "Thread 2");
        Thread t3 = new Thread(new TaskCyclic(cb), "Thread 3");
    }
}

/* Concurrent Collections */
public class CopyDemonstration {
    public static void main(String[] args) throws InterruptedException {
        //regular array
        ArrayList<Integer> array_list = new ArrayList<>();
        array_list.ensureCapacity(500000);
        //initialize a new CopyOnwrite ArrayList with 500,000 numbers
        CopyOnWriteArrayList<Integer> numbers = new CopyOnWriteArrayList<>(array_list);

        //calculate the time it takes to add a number in CoW AL
        long startTime = System.nanoTime();
        numbers.add(500001);
        long endTime = System.nanoTime();
        long duration = (endTime - startTime);

        //regular AL
        long startTime_al = System.nanoTime();
        array_list.add(500001);
        long endTime_al = System.nanoTime();
        long duration_al = (endTime_al - startTime_al);

        System.out.println("Time taken by a regular arrayList: " + duration_al + " nano seconds");
        System.out.println("Time taken by a CopyOnWrite arrayList: " + duration + " nano seconds");
    }
}