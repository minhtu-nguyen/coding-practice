// *** CREATIONAL
// Builder
public abstract class AircraftBuilder {

    public void buildEngine() {

    }

    public void buildWings() {

    }

    public void buildCockpit() {

    }

    public void buildBathrooms() {

    }

    abstract public IAircraft getResult();
}

public class Boeing747Builder extends AircraftBuilder {

    Boeing747 boeing747;

    @Override
    public void buildCockpit() {

    }

    @Override
    public void buildEngine() {

    }

    @Override
    public void buildBathrooms() {
        
    }

    @Override
    public void buildWings() {

    }

    public IAircraft getResult() {
        return boeing747;
    }
}

public class F16Builder extends AircraftBuilder {

    F16 f16;

    @Override
    public void buildEngine() {
        // get F-16 an engine
        // f16.engine = new F16Engine();
    }

    @Override
    public void buildWings() {
        // get F-16 wings
        // f16.wings = new F16Wings();
    }

    @Override
    public void buildCockpit() {
        f16 = new F16();
        // get F-16 a cockpit
        // f16.cockpit = new F16Cockpit();
    }

    public IAircraft getResult() {
        return f16;
    }
}

public class Director {

    AircraftBuilder aircraftBuilder;

    public Director(AircraftBuilder aircraftBuilder) {
        this.aircraftBuilder = aircraftBuilder;
    }

    public void construct(boolean isPassenger) {
        aircraftBuilder.buildCockpit();
        aircraftBuilder.buildEngine();
        aircraftBuilder.buildWings();

        if (isPassenger)
            aircraftBuilder.buildBathrooms();
    }
}

public class Client {

    public void main() {

        F16Builder f16Builder = new F16Builder();
        Director director = new Director(f16Builder);
        director.construct(false);

        IAircraft f16 = f16Builder.getResult();
    }
}

// Singleton
public class AirforceOne {

    // The sole instance of the class
    private static AirforceOne onlyInstance;

    // Make the constructor private so its only accessible to
    // members of the class.
    private AirforceOne() {
    }

    public void fly() {
        System.out.println("Airforce one is flying...");
    }

    // Create a static method for object creation
    public static AirforceOne getInstance() {

        // Only instantiate the object when needed.
        if (onlyInstance == null) {
            onlyInstance = new AirforceOne();
        }

        return onlyInstance;
    }
}

public class Client {

    public void main() {
        AirforceOne airforceOne = AirforceOne.getInstance();
        airforceOne.fly();
    }
}

public class AirforceOneWithDoubleCheckedLocking {

    // The sole instance of the class. Note its marked volatile
    private volatile static AirforceOneWithDoubleCheckedLocking onlyInstance;

    // Make the constructor private so its only accessible to
    // members of the class.
    private AirforceOneWithDoubleCheckedLocking() {
    }

    public void fly() {
        System.out.println("Airforce one is flying...");
    }

    // Create a static method for object creation
    synchronized public static AirforceOneWithDoubleCheckedLocking getInstance() {

        // Only instantiate the object when needed.
        if (onlyInstance == null) {
            // Note how we are synchronizing on the class object
            synchronized (AirforceOneWithDoubleCheckedLocking.class) {
                if (onlyInstance == null) {
                    onlyInstance = new AirforceOneWithDoubleCheckedLocking();
                }
            }
        }

        return onlyInstance;
    }
}

// Prototype
public interface IAircraftPrototype {

    void fly();

    IAircraftPrototype clone();

    void setEngine(F16Engine f16Engine);
}

public class F16 implements IAircraftPrototype {

    // default engine
    F16Engine f16Engine = new F16Engine();

    @Override
    public void fly() {
        System.out.println("F-16 flying...");
    }

    @Override
    public IAircraftPrototype clone() {
        // Deep clone self and return the product
        return new F16();
    }

    public void setEngine(F16Engine f16Engine) {
        this.f16Engine = f16Engine;
    }
}

public class Client {

    public void main() {

        IAircraftPrototype prototype = new F16();

        // Create F16-A
        IAircraftPrototype f16A = prototype.clone();
        f16A.setEngine(new F16AEngine());

        // Create F16-B
        IAircraftPrototype f16B = prototype.clone();
        f16B.setEngine(new F16BEngine());
    }
}

// Factory
public class F16 {

    IEngine engine;
    ICockpit cockpit;

    protected F16 makeF16() {
        engine = new F16Engine();
        cockpit = new F16Cockpit();
        return this;
    }

    public void taxi() {
        System.out.println("F16 is taxing on the runway !");
    }

    public void fly() {
        // Note here carefully, the superclass F16 doesn't know
        // what type of F-16 variant it was returned.
        F16 f16 = makeF16();
        f16.taxi();
        System.out.println("F16 is in the air !");
    }
}

public class F16A extends F16 {

    @Override
    public F16 makeF16() {
        super.makeF16();
        engine = new F16AEngine();
        return this;
    }
}

public class F16B extends F16 {

    @Override
    public F16 makeF16() {
        super.makeF16();
        engine = new F16BEngine();
        return this;
    }
}

public class Client {
    public void main() {
        Collection<F16> myAirForce = new ArrayList<F16>();
        F16 f16A = new F16A();
        F16 f16B = new F16B();
        myAirForce.add(f16A);
        myAirForce.add(f16B);

        for (F16 f16 : myAirForce) {
            f16.fly();
        }
    }
}

// Abstract Factory
public interface IAircraftFactory {

    IEngine createEngine();

    IWings createWings();

    ICockpit createCockpit();
}

public class F16Factory implements IAircraftFactory {

    @Override
    public IEngine createEngine() {
        return new F16Engine();
    }

    @Override
    public IWings createWings() {
        return new F16Wings();
    }

    @Override
    public ICockpit createCockpit() {
        return new F16Cockpit();
    }
}

public class Boeing747Factory implements IAircraftFactory {

    @Override
    public IEngine createEngine() {
        return new Boeing747Engine();
    }

    @Override
    public IWings createWings() {
        return new Boeing747Wings();
    }

    @Override
    public ICockpit createCockpit() {
        return new Boeing747Cockpit();
    }
}

//Complete
public class Aircraft {

    IEngine engine;
    ICockpit cockpit;
    IWings wings;
    IAircraftFactory factory;

    public Aircraft(IAircraftFactory factory) {
        this.factory = factory;
    }

    protected Aircraft makeAircraft() {
        engine = factory.createEngine();
        cockpit = factory.createCockpit();
        wings = factory.createWings();
        return this;
    }

    private void taxi() {
        System.out.println("Taxing on runway");
    }

    public void fly() {
        Aircraft aircraft = makeAircraft();
        aircraft.taxi();
        System.out.println("Flying");
    }
}

public class Client {

    public void main() {

        // Instantiate a concrete factory for F-16
        F16Factory f16Factory = new F16Factory();
        
        // Instantiate a concrete factory for Boeing-747
        Boeing747Factory boeing747Factory = new Boeing747Factory();
        
        // Lets create a list of all our airplanes
        Collection<Aircraft> myPlanes = new ArrayList<>();
        
        // Create a new F-16 by passing in the f16 factory
        myPlanes.add(new Aircraft(f16Factory));

        // Create a new Boeing-747 by passing in the boeing factory
        myPlanes.add(new Aircraft(boeing747Factory));

        // Fly all your planes
        for (Aircraft aircraft : myPlanes) {
            aircraft.fly();
        }

    }
}

// *** STRUCTURAL
// Adapter
public class HotAirBalloon {

    String gasUsed = "Helium";

    void fly(String gasUsed) {
        // Take-off sequence based on the kind of feul
        // Followed by more code.  
    }

    // Function returns the gas used by the balloon for flight
    String inflateWithGas() {
        return gasUsed;
    }
}

public interface IAircraft {
    void fly();
}

public class Adapter implements IAircraft {

    HotAirBalloon hotAirBalloon;

    public Adapter(HotAirBalloon hotAirBalloon) {
        this.hotAirBalloon = hotAirBalloon;
    }

    @Override
    public void fly() {
        String feulUsed = hotAirBalloon.inflateWithGas();
        hotAirBalloon.fly(feulUsed);
    }
}

public void main() {

    HotAirBalloon hotAirBalloon = new HotAirBalloon();
    Adapter hotAirBalloonAdapter = new Adapter(hotAirBalloon);
    
    hotAirBalloonAdapter.fly();
}

// Bridge
public abstract class AbstractCorollaImpl {

    abstract void listSafetyEquipment();

    abstract boolean isCarRightHanded();
}

public abstract class AbstractCorolla {

    protected AbstractCorollaImpl corollaImpl;

    public AbstractlCorolla(AbstractCorollaImpl corollaImpl) {
        this.corollaImpl = corollaImpl;
    }

    abstract void listSafetyEquipment();

    abstract boolean isCarRightHanded();
}

public class Corolla_L extends AbstractCorolla {

    public Corolla_L(AbstractCorollaImpl corollaImpl) {
        super(corollaImpl);
    }

    @Override
    void listSafetyEquipment() {
        corollaImpl.listSafetyEquipment();

    }

    @Override
    boolean isCarRightHanded() {
        return corollaImpl.isCarRightHanded();
    }
}

public class Corolla_L_Impl_AsiaPacific extends AbstractCorollaImpl {

    @Override
    void listSafetyEquipment() {
        System.out.println("Not so safe.");
    }

    @Override
    boolean isCarRightHanded() {
        return false;
    }
}

public class Corolla_L_Impl_NorthAmerica extends AbstractCorollaImpl {

    @Override
    void listSafetyEquipment() {
        System.out.println("High safety standards.");
    }

    @Override
    boolean isCarRightHanded() {
        return true;
    }
}

public class Client {

    public void main() {

        AbstractCorolla myCorolla = new Corolla_L(new Corolla_L_Impl_AsiaPacific());
        System.out.println(myCorolla.isCarRightHanded());

        myCorolla.setCorollaImpl(new Corolla_L_Impl_NorthAmerica());
        System.out.println(myCorolla.isCarRightHanded());
    }
}

// Composite
public interface IAlliancePart {

    // Any implementing class should return the
    // the number of personnel or staff required
    // to operate the aircraft or the airfoce
    int getPersonnel();
}

public class F16 implements IAircraft, IAlliancePart {

    @Override
    public int getPersonnel() {
        // We need 2 pilots for F-16
        return 2;
    }
}

public class C130Hercules implements IAircraft, IAlliancePart {

    @Override
    public int getPersonnel() {
        // This cargo plane, needs 5 people
        return 5;
    }
}

public class Airforce implements IAlliancePart {

    ArrayList<IAlliancePart> planes = new ArrayList<>();

    public void add(IAlliancePart alliancePart) {
        planes.add(alliancePart);
    }

    @Override
    public int getPersonnel() {

        // We iterator over the entire air force which can
        // contain leaf nodes (planes) or other composites
        // (air forces). This iteration is an example of an
        // internal iterator.
        Iterator<IAlliancePart> itr = planes.iterator();
        int staff = 0;

        while (itr.hasNext()) {
            staff += itr.next().getPersonnel();
        }

        return staff;
    }
}

public class Client {

    public void main() {

        Airforce NatoAllaiance = new Airforce();
        
        // The nested methods aren't listed for brevity's sake
        NatoAllaiance.add(createCanadaAirForce());
        NatoAllaiance.add(createUSAAirForce());

        F16 frenchF16 = new F16();
        C130Hercules germanCargo = new C130Hercules();

        NatoAllaiance.add(frenchF16);
        NatoAllaiance.add(germanCargo);

        System.out.println(NatoAllaiance.getPersonnel());
    }
}

// Decorator
public interface IAircraft {

    float baseWeight = 100;

    void fly();

    void land();

    float getWeight();

}

public class Boeing747 implements IAircraft {

    @Override
    public void fly() {
        System.out.println("Boeing-747 flying ...");
    }

    @Override
    public void land() {
        System.out.println("Boeing-747 landing ...");
    }

    @Override
    public float getWeight() {
        return baseWeight;
    }
}

public abstract class BoeingDecorator implements IAircraft {

}

public class LuxuryFittings extends BoeingDecorator {

    IAircraft boeing;

    public LuxuryFittings(IAircraft boeing) {
        this.boeing = boeing;
    }

    @Override
    public void fly() {
        boeing.fly();
    }

    @Override
    public void land() {
        boeing.land();
    }

    @Override
    public float getWeight() {
        return (30.5f + boeing.getWeight());
    }
}

public class BulletProof extends BoeingDecorator {

    IAircraft boeing;

    public BulletProof(IAircraft boeing) {
        this.boeing = boeing;

    }

    @Override
    public void fly() {
        boeing.fly();
    }

    @Override
    public void land() {
        boeing.land();
    }

    @Override
    public float getWeight() {
        return 50f + boeing.getWeight();
    }
}

public class Client {

    public void main() {
        IAircraft simpleBoeing = new Boeing747();
        IAircraft luxuriousBoeing = new LuxuryFittings(simpleBoeing);
        IAircraft bulletProofBoeing = new BulletProof(luxuriousBoeing);
        float netWeight = bulletProofBoeing.getWeight();
        System.out.println("Final weight of the plane: " + netWeight);
    }
}

// Facade
public class AutopilotFacade {

    private BoeingAltitudeMonitor altitudeMonitor;
    private BoeingEngineController engineController;
    private BoeingFuelMonitor feulMonitor;
    private BoeingNavigationSystem navigationSystem;

    public AutopilotFacade(BoeingAltitudeMonitor altitudeMonitor, BoeingEngineController engineController,
                           BoeingFuelMonitor feulMonitor, BoeingNavigationSystem navigationSystem) {
        this.altitudeMonitor = altitudeMonitor;
        this.engineController = engineController;
        this.feulMonitor = feulMonitor;
        this.navigationSystem = navigationSystem;
    }

    public void autopilotOn() {
        altitudeMonitor.autoMonitor();
        engineController.setEngineSpeed(700);
        navigationSystem.setDirectionBasedOnSpeedAndFeul(
                engineController.getEngineSpeed(),
                feulMonitor.getRemainingFeulInGallons());
    }

    public void autopilotOff() {
        altitudeMonitor.turnOff();
        engineController.turnOff();
        navigationSystem.turnOff();
        feulMonitor.turnOff();
    }
}

// Flyweight
public class F16 implements IAircraft {

    // Intrinsic state
    private final String name = "F16";
    private final int personnel = 2;
    private final String dimensions = "15m long 3m wide";
    private final String wingspan = "33 feet";

    // Extrinsic state includes the current position and current speed
    // of the aircraft that is being passed in for computing remaining
    // time to destination
    public double getTimeToDestination(int currX, int currY, int destX, int destY, int currSpeed) {

        // algorithm to calculate the remaining time to reach
        // destination.

        return 1;
    }
}

public class Client {

    public void main(int[][] coordsF16) {

        F16 flyweightF16 = new F16();

        for (int i = 0; i < coordsF16.length; i++) {
            int currX = coordsF16[i][0];
            int currY = coordsF16[i][1];

            // We are passing in the extrinsic state to the flyweight object. Note we are storing the
            // extrinsic state of the airborne f16s in a 2-dimensional array.
            System.out.println("time to destination = " +
                               flyweightF16.getTimeToDestination(currX, currY, 10, 10, 200));
        }

    }
}

// Proxy
public interface IDrone {

    void turnLeft();

    void turnRight();

    void firstMissile();
}

public class DroneProxy implements IDrone {

    @Override
    public void turnLeft() {
        // forward request to the real drone to
        // turn left over the internet
    }

    @Override
    public void turnRight() {
        // forward request to the real drone to
        // turn right over the internet
    }

    @Override
    public void firstMissile() {
        // forward request to the real drone to
        // fire missile
    }
}

public class Client {

    public void main(DroneProxy droneProxy) {

        // perpetual loop that received pilot actions
        while (true) {

            Scanner scanner = new Scanner(System.in);
            String action = scanner.nextLine();

            switch (action) {
            case "left": {
                droneProxy.turnLeft();
                break;
            }

            case "right": {
                droneProxy.turnRight();
                break;
            }

            case "fire": {
                droneProxy.firstMissile();
                break;
            }

            default:
                System.out.println("Invalid Action");
            }
        }
    }
}

public class Drone implements IDrone {

    @Override
    public void turnLeft() {
        // receives the request and any method parameters
        // over the internet to turn the drone to its left.
    }

    @Override
    public void turnRight() {
        // receives the request and any method parameters
        // over the internet to turn the drone to its right.
    }

    @Override
    public void firstMissile() {
        // receives the request and any method parameters
        // over the internet to fire a missile
    }
} 

// *** BEHAVIORAL
// Chain of Responsibility
abstract public class AbstractRequest {

    // Each request is identified by a an integer
    // FireRequest: 1
    // LowFuelRequest: 2
    private int requestCode;

    public AbstractRequest(int requestCode) {
        this.requestCode = requestCode;
    }

    public int getRequestCode() {
        return requestCode;
    }
}

abstract public class AbstractHandler {

    private AbstractHandler next;

    public AbstractHandler(AbstractHandler next) {
        this.next = next;
    }

    public void setNext(AbstractHandler next) {
        this.next = next;
    }

    public void handleRequest(AbstractRequest request) {
        if (next != null) {
            next.handleRequest(request);
        }
    }
}

public class FireDetectedRequest extends AbstractRequest {

    // Fire request is assigned code of 1
    public FireDetectedRequest() {
        super(1);
    }
}

public class LowFuelRequest extends AbstractRequest {

    // Low on fuel request is assigned code of 2
    public LowFuelRequest() {
        super(2);
    }
}

public class FireHandler extends AbstractHandler {

    // Only interested in handling requests with code 1
    private static int code = 1;

    public FireHandler(AbstractHandler successor) {
        super(successor);
    }

    @Override
    public void handleRequest(AbstractRequest request) {
        if (code == request.getRequestCode()) {
            // Handle the request here.
        } else {
            // If the handler, doesn't handle these type of
            // requests, it can just call the super class's
            // forward request method.
            super.handleRequest(request);
        }
    }
}

public class Client {

    public void main() {

        // Setup the chain like so:
        // FireHandler --> LowFuelHandler --> null
        // The chain has just two handlers with the firstHandler
        // at the head of the chain.
        AbstractHandler lowFuelHandler = new LowFuelHandler(null);
        FireHandler fireHandler = new FireHandler(lowFuelHandler);

        // Create a emergency request that the airplane is running low
        // on fuel.
        LowFuelRequest lowFuelRequest = new LowFuelRequest();

        // Let the chain handle the request
        fireHandler.handleRequest(lowFuelRequest);
    }
}

// Observer
public interface ISubject {

    void addObserver(IObserver observer);

    void removeObserver(IObserver observer);

    void notifyObservers();
}

public interface IObserver {

    void update( Object newState);
}

public class ControlTower implements ISubject {

    // The ControlTower maintains a list of
    Collection<IObserver> observers = new ArrayList<>();

    @Override
    public void addObserver(IObserver observer) {
        observers.add(observer);

    }

    @Override
    public void removeObserver(IObserver observer) {
        // Logic to remove the observer goes in here
    }

    @Override
    public void notifyObservers() {
        for (IObserver observer : observers) {
            // We are passing null for state here but we
            // could pass 'this' the subject itself or a
            // type representing the state. These two options
            // represent the Pull vs Push models
            observer.update( null);
        }
    }

    /**
     * This is hypothetical function that runs perptually, gathering
     * runway and weather conditions and notifying all observers of
     * them periodically.
     */
    public void run() {

        while (true) {
            // get new runway/weather conditions and update observers
            // every five minutes
            // Thread.sleep(1000 * 60 * 5)
            notifyObservers();
        }
    }
}

public class F16 implements IObserver, IAircraft {

    ISubject observable;

    public F16(ISubject observable) {
        this.observable = observable;
        observable.addObserver(this);
    }

    @Override
    public void fly() {
        System.out.println("F16 is flying ...");

    }

    @Override
    public void land() {

        // Don't forget to unsubscribe from control tower updates
        observable.removeObserver(this);
    }

    @Override
    public void update(Object newState) {
        // Take appropriate action based on newState
    }
}

// Interpreter
public abstract class Program {
    public void interpret(Context context) {}
}

public abstract class Flight extends Program {

    Flight flight;
    ShowOff showOff;
    Glide glide;

    @Override
    public void interpret(Context context) {}
}

public class ShowOff extends Program {

    BarrelRoll barrelRoll;
    SplitS splitS;

    @Override
    public void interpret(Context context) {

    }
}

public class BarrelRoll extends Flight {

    @Override
    public void interpret(Context context) {

    }
}

public class Glide extends Program {

    @Override
    public void interpret(Context context) {

    }
}

public class SplitS extends Program {

    @Override
    public void interpret(Context context) {

    }
}

public class Client {

    public void main(AbstractSyntaxTree ast) {

        Context context = new Context("glide splitS glide barelRoll glide");

        while (ast.hasNext()) {
            Program node = ast.getNextNode();
            node.interpret(context);
        }
    }
}

// Command
public interface Command {
    void execute();
}

public class LandingGearDownCommand implements Command {

    // This is called the receiver of the request and
    // actually has the logic to perform the action
    LandingGear landingGear;

    public LandingGearDownCommand(LandingGear landingGear) {
        this.landingGear = landingGear;
    }

    @Override
    public void execute() {
        landingGear.up();
    }
}

public class InstrumentPanel {

    // Only two commands for now
    Command[] commands = new Command[2];

    public InstrumentPanel() {

    }

    public void setCommand(int i, Command command) {
        commands[i] = command;
    }

    public void lowerLandingGear() {
        // Assuming that the client correctly sets the first
        // index to be the landing gear lower command
        commands[0].execute();
    }

    public void retractLandingGear() {
        commands[1].execute();
    }
}

public void main() {

    LandingGear landingGear = new LandingGear();
    LandingGearDownCommand landingGearDownCommand = new LandingGearDownCommand(landingGear);
    LandingGearUpCommand landingGearUpCommand = new LandingGearUpCommand(landingGear);

    // Create the instrument panel
    InstrumentPanel instrumentPanel = new InstrumentPanel();

    // Assign the commands
    instrumentPanel.setCommand(0, landingGearUpCommand);
    instrumentPanel.setCommand(1, landingGearDownCommand);

    // Lower the landing gear
    instrumentPanel.lowerLandingGear();

    // Retract the landing gear
    instrumentPanel.retractLandingGear();
}

// Iterator Pattern
public interface Iterator {

    void next();

    boolean hasNext();
}

public class AirForce {

    List<IAircraft> jets = new ArrayList<>();
    IAircraft[] helis = new IAircraft[1];
    LinkedList<Boeing747> cargo = new LinkedList<>();

    public List<IAircraft> getJets() {
        return jets;
    }

    public IAircraft[] getHelis() {
        return helis;
    }

    public LinkedList<Boeing747> getCargo() {
        return cargo;
    }

    public AirForce() {
        jets.add(new F16());
        helis[0] = new CobraGunship();
        cargo.add(new Boeing747());
    }

    // This method returns a concrete iterator that
    // traverses over the entire airforce planes.
    public Iterator createIterator() {
        return new AirForceIterator(this);

    }

    // This method returns a different concerete iterator
    // that traverses over only the jets in the airforce.
    public Iterator createJetsIterator() {
        return new JetsIterator(jets);

    }
}

public class AirForceIterator implements Iterator {

    List<IAircraft> jets;
    IAircraft[] helis;
    LinkedList<Boeing747> cargo;
    int jetsPosition = 0;
    int helisPosition = 0;
    int cargoPosition = 0;

    /**
     * The iterator is composed with the object it'll be iterating over
     */
    public AirForceIterator(AirForce airForce) {
        jets = airForce.getJets();
        helis = airForce.getHelis();
        cargo = airForce.getCargo();

    }

    @Override
    /**
     * We provide our own custom logic of returning aircraft in a
     * sequence. Note we are returning IAircraft interface object which
     * every plane in our airforce implements. We also design the function
     * to throw a runtime exception if next is invoked when no more elements
     * are left to iterate over
     */
    public IAircraft next() {

        // return helis first
        if (helisPosition < helis.length) {
            return helis[helisPosition++];
        }

        // return jets second
        if (jetsPosition < jets.size()) {
            return jets.get(jetsPosition++);
        }

        // return cargos last
        if (cargoPosition < cargo.size()) {
            return cargo.get(cargoPosition++);
        }

        throw new RuntimeException("No more elements");

    }

    @Override
    public boolean hasNext() {

        return helis.length > helisPosition ||
               jets.size() > jetsPosition ||
               cargo.size() > cargoPosition;
    }
}

public class Client {

    public void main() {

        AirForce airForce = new AirForce();

        Iterator jets = airForce.createJetsIterator();

        while (jets.hasNext()) {
            jets.next();
        }

        Iterator allPlanes = airForce.createIterator();

        while (allPlanes.hasNext()) {
            allPlanes.next();
        }
    }
}

// Mediator Pattern
public class ControlTower {

    List<IAircraft> queuedForLanding = new ArrayList<>();

    // An aircraft just notifies the control tower that it wants to
    // land and doesn't coordinate with other aircraft
    synchronized public void requestToLand(IAircraft aircraft) {
        queuedForLanding.add(aircraft);
    }

    public void run() {

        // perpetual loop
        while (true) {

            // inefficient busy wait till aircraft requests to land
            while(queuedForLanding.size() == 0);

            IAircraft aircraft;
            synchronized (this) {
                aircraft = queuedForLanding.remove(0);
            }
            // We have only one runway available so only allow a single
            // aircraft to land.
            aircraft.land();
        }
    }
}

public interface IAircraft {

    public void land();
}

public class F16 implements IAircraft {

    ControlTower controlTower;

    public F16(ControlTower controlTower) {
        this.controlTower = controlTower;
    }

    @Override
    public void land() {
        System.out.println("F16 landing...");
    }

    public void requestControlTowerToLand() {
        controlTower.requestToLand(this);

    }
}

// Memento Pattern
public class BlackBox implements Serializable {

    private long altitude;
    private double speed;
    private float engineTemperature;
    private static final long serialVersionUID = 1L;

    public BlackBox(long altitude, double speed, float engineTemperature) {
        this.altitude = altitude;
        this.speed = speed;
        this.engineTemperature = engineTemperature;
    }

    // Saving the state of the object as a byte stream
    public byte[] getState() throws IOException {
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        ObjectOutput out = null;
        byte[] memento = null;
        try {
            out = new ObjectOutputStream(bos);
            out.writeObject(this);
            out.flush();
            memento = bos.toByteArray();
        } finally {
            try {
                bos.close();
            } catch (IOException ex) {
                // ignore close exception
            }
        }
        return memento;
    }

    // Restoring state from memento
    public BlackBox setState(byte[] memento) throws Exception {
        ByteInputStream bis = new ByteInputStream(memento, memento.length);

        ObjectInputStream objectInputStream
                = new ObjectInputStream(bis);
        BlackBox blackBox = (BlackBox) objectInputStream.readObject();
        objectInputStream.close();
        return blackBox;
    }

}

public class Client {

    public void main(BlackBox blackBox) throws Exception{

        // Save the state of the memento as a byte stream.
        byte[] memento = blackBox.getState();

        // Do some work.

        // Now restore the blackbox to the previous state
        blackBox = blackBox.setState(memento);

    }

}

// State Pattern
/**
 * This interface defines the actions a pilot
 * can take against the aircraft object. Each
 * action will move the aircraft into a different
 * state
 */
public interface IAircraft {
   //Empty interface
}

public class F16 implements IAircraft {

    private ParkedState parkedState = new ParkedState(this);
    private CrashState crashState = new CrashState(this);
    private LandState landState = new LandState(this);
    private TaxiState taxiState = new TaxiState(this);
    private AirborneState airborneState = new AirborneState(this);

    IPilotActions state;

    public F16() {
        state = parkedState;
    }

    void startsEngine() {
        state.pilotTaxies(this);
    }

    void fliesPlane() {
        state.pilotFlies(this);
    }

    void landsPlane() {
        state.pilotLands(this);
    }

    void ejectsPlane() {
        state.pilotEjects(this);
    }

    void parksPlane() {
        state.pilotParks(this);
    }

    void setState(IPilotActions IPilotActions) {
        state = IPilotActions;
    }

    ParkedState getParkedState() {
        return parkedState;
    }

    CrashState getCrashState() {
        return crashState;
    }

    LandState getLandState() {
        return landState;
    }

    TaxiState getTaxiState() {
        return taxiState;
    }

    public AirborneState getAirborneState() {
        return airborneState;
    }

}

public interface IPilotActions {

    void pilotTaxies(F16 f16);

    void pilotFlies(F16 f16);

    void pilotEjects(F16 f16);

    void pilotLands(F16 f16);

    void pilotParks(F16 f16);
}

public class ParkedState implements IPilotActions {

    F16 f16;

    // Notice, how the state class is composed with the context object
    public ParkedState(F16 f16) {
        this.f16 = f16;
    }

    @Override
    public void pilotTaxies(F16 f16) {
        f16.setState(f16.getTaxiState());
    }

    @Override
    public void pilotFlies(F16 f16) {
        System.out.println("This is an invalid operation for this state");
    }

    @Override
    public void pilotEjects(F16 f16) {
        f16.setState(f16.getCrashState());

    }

    @Override
    public void pilotLands(F16 f16) {
        System.out.println("This is an invalid operation for this state");
    }

    @Override
    public void pilotParks(F16 f16) {
        System.out.println("This is an invalid operation for this state");
    }
}

public class Client {

    public void main() {

        F16 f16 = new F16();
        f16.startsEngine();
        f16.fliesPlane();
        f16.ejectsPlane();
        
    }
}


// Template Method
public abstract class AbstractPreFlightCheckList {

    // This method captures the template or the skeleton
    // of the algorithm for the pre-flight checklist.
    final public void runChecklist() {

        // Check fuel guage
        IsFuelEnough();

        // Check doors are locked
        doorsLocked();

        // Check air pressure
        checkAirPressure();
    }

    // Don't let subclasses override this method, this is a
    // mandatory check
    final protected void IsFuelEnough() {
        System.out.println("check fuel gauge");
    }

    // Some airplanes may or may not have doors so allow this
    // method to be overridden by subclasses
    protected boolean doorsLocked() {
        return true;
    }

    // Force subclasses to provide their own way of checking for
    // cabin or cockpit air pressure
    abstract void checkAirPressure();
}

public class F16PreFlightCheckList extends AbstractPreFlightCheckList {

    @Override
    void checkAirPressure() {
        // Implement the custom logic for checking cockpit
        // air pressure for F-16
    }

    @Override
    protected boolean doorsLocked() {
        // F-16 unlike a Boeing-747 has no doors
        // so always return true;
        return true;
    }
}

// Strategy Pattern
public interface ISort {

    void sort(int[] input);
}

public class BubbleSort implements ISort {

    @Override
    public void sort(int[] input) {
        // Do inefficient sorting in order n squared
    }
}

public class MergeSort implements ISort {

    @Override
    public void sort(int[] input) {
        // Do efficient sorting in nlogn
    }
}

public class Context {

    private ISort howDoISort;

    public Context(ISort howDoISort) {
        this.howDoISort = howDoISort;
    }

    // Context receives the data from its client
    // and passes it on to the strategy object.
    void sort(int[] numbers) {
        howDoISort.sort(numbers);
    }

    // We can change the sorting algorithm using this setter
    public void setHowDoISort(ISort howDoISort) {
        this.howDoISort = howDoISort;
    }
}

public class Client {

    private int[] numbers = new int[1000];

    void crunchingNumbers() {

        // Choose the sorting strategy
        BubbleSort bubbleSort = new BubbleSort();

        // Context receives the strategy object
        Context context = new Context(bubbleSort);

        // Sort the numbers
        context.sort(numbers);

        // Do remaining work
    }
}

// Visitor Pattern
public class Airforce {

    // Holds a collection of planes
    private Collection<IAircraft> planes = new ArrayList<>();

    // Returns an iterator to its collection of planes
    public Iterator<IAircraft> getIterator() {
        return planes.iterator();
    }
}

public interface IAircraft {

    // Each concrete element class is expected to
    // define the accept method
    public void accept(IAircraftVisitor visitor);

}

public class F16 implements IAircraft {

    @Override
    public void accept(IAircraftVisitor visitor) {
        visitor.visitF16(this);
    }
}

public class Boeing747 implements IAircraft{

    @Override
    public void accept(IAircraftVisitor visitor) {
        visitor.visitBoeing747(this);
    }
}

public interface IAircraftVisitor {

    void visitF16(F16 f16);

    void visitBoeing747(Boeing747 boeing747);
}

public class MetricsVisitor implements IAircraftVisitor {

    public void visitF16(F16 f16){
        // Logic to get metrics for F16
    }

    public void visitBoeing747(Boeing747 boeing747){
        // Logic to get metrics for Boeing 747
    }

    public void printAccumulatedResults(){
        
    }
}

public class PriceVisitor implements IAircraftVisitor{

    @Override
    public void visitF16(F16 f16) {
        // Logic to get price for F16
    }

    @Override
    public void visitBoeing747(Boeing747 boeing747) {
        // Logic to get price for Boeing 747
    }

    public void printAccumulatedResults(){
        
    }
}

public class Client {

    public void main(Airforce airforce) {

        Iterator<IAircraft> planes = airforce.getIterator();
        MetricsVisitor aircraftVisitor = new MetricsVisitor();

        while (planes.hasNext()){
            planes.next().accept(aircraftVisitor);
        }
        
        aircraftVisitor.printAccumulatedResults();
    }
}