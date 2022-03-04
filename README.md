# JD-GOC-VRP
[点击进入赛题详情...](https://jdata.jd.com/html/detail.html?id=5)

VRPTW variation of JD Global Optimization Challange (2018/9/15, Final rank: 15/1921)


Algorithm summary:
1. Generate an initial solution based on Time-oriented Nearest-Neighborhood Heuristic proposed by Solomon(1987).
2. Improve the initial solution using Large Neighbourhood Search algorithm and Simulated Annealing algorithm. We used 7 operators to search neighborhood solution. The following two papers are listed for reference:<br>
    [--A Two-Stage Hybrid Local Search for the Vehicle Routing Problem with Time Windows](https://pdfs.semanticscholar.org/c88a/7ae45e8905a674e09a543b7228a6190c9d92.pdf)<br>
    [--An advanced hybrid meta-heuristic algorithm for the vehicle routing problem with backhauls and time windows](https://www.sciencedirect.com/science/article/abs/pii/S0360835214003453)
3. Other implementation skills.


Code is just for reference.
