import pandas as pd
import numpy as np
from copy import copy
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from tqdm.notebook import tqdm
from geopy.distance import geodesic
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import heapq

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Для гистограммы

import solara
from matplotlib.figure import Figure
from mesa.visualization.utils import update_counter

import mesa
from agents import (
    Order,
    Restaurant,
    Customer,
    Courier,
    time_moving,
    SelectiveDataCollector,
)

from scipy.stats import skewnorm


def proportion_of_waiting_orders(model):
    """
    Метод для отслеживания количества ожидающих заказов
    """
    number_of_waiting_orders = len(
        [
            agent
            for agent in model.agents
            if isinstance(agent, Order) and not agent.picked_up
        ]
    )
    return number_of_waiting_orders


class DeliveryModel(mesa.Model):

    def __init__(
        self,
        data: pd.DataFrame = None,
        couriers: list = None,
        use_nearest: bool = False,
        priority: str = None,
        aggregation: str = None,
        num_orders: int = 10,
        num_couriers: int = None,
        max_time: float = 10,
        num_rest: int = 5,
        num_customers: int = 5,
        peak_times: list = [2, 7],
        peak_intensity: float = 3.0,
        scale: float = 1.0,
        res_scale: list = [0.04, 0.05],
        customer_scale: list = [0.03, 0.04],
        portion_customers_normal: float = 0.5,
        city_center: tuple = (0, 0),
        left_quantile: float = 0.15,
        right_quantile: float = 0.85,
        eps: float = None,
        random_state: int = 42,
        verbose: bool = False,
        dummy_movement=True,
        pre_trained_model_predict_p_i=None,
        speed=1,
        debug=False,
        no_map=False,
        free_courier_movement=False,
        max_distance=150,
    ):
        super().__init__(seed=random_state)

        if data is None:
            # вызов генератора данных
            self.data = self.data_generation(
                num_orders,
                max_time,
                num_rest,
                num_customers,
                peak_times,
                peak_intensity,
                scale,
                city_center,
                random_state,
                res_scale,
                customer_scale,
                portion_customers_normal,
            )
        else:
            self.data = data

        if couriers is None:
            if num_couriers is not None:
                # вызов генератора курьеров
                self.couriers = self.courier_generation(
                    self.data, num_couriers, left_quantile, right_quantile
                )
                self.num_couriers = num_couriers
            else:
                raise ValueError(
                    "Нет ни курьеров, ни количества курьеров для их генерации"
                )
        else:
            self.couriers = copy(couriers)
            self.num_couriers = len(couriers)

        self.use_nearest = use_nearest
        self.aggregation = aggregation
        self.priority = priority
        self.free_courier_movement = free_courier_movement
        if isinstance(speed, int) or isinstance(speed, float):
            self.speed = [speed] * self.num_couriers
        elif isinstance(speed, list):
            assert len(speed) == self.num_couriers
            self.speed = speed
        else:
            raise ValueError("Неправильно задана скорость")
        if self.aggregation is None:
            if self.priority not in [
                None,
                "Deadline",
                "reserve",
                "fastest",
                "longest",
                "ML_fastest",
                "ML_longest",
            ]:
                raise ValueError("Не совместимый тип priority и aggregation")

        if eps is None:
            coords = self.data[
                [
                    "Res_lat",
                    "Res_long",
                    "Del_lat",
                    "Del_long",
                ]
            ]
            self.eps = distance_matrix(coords.values, coords.values).mean()
        else:
            self.eps = eps
        self.random_state = random_state
        self.max_distance = max_distance

        # data generation
        self.num_orders = num_orders
        self.max_time = max_time
        self.num_rest = num_rest

        # environment generation
        self.verbose = verbose
        self.dummy_movement = dummy_movement
        self.served_orders = 0
        self.debug = debug
        self.no_map = no_map

        # ML
        if pre_trained_model_predict_p_i:
            self.pre_trained_model_predict_p_i = pre_trained_model_predict_p_i

        # Создаем пространство для перемещения
        if not self.no_map:
            self.space = mesa.space.ContinuousSpace(
                self.data[["Del_lat", "Res_lat"]].max().max() + 2,
                self.data[["Del_long", "Res_long"]].max().max() + 2,
                True,
                self.data[["Del_lat", "Res_lat"]].min().min() - 2,
                self.data[["Del_long", "Res_long"]].min().min() - 2,
            )
            self.long_centr = self.data[["Del_long", "Res_long"]].mean().mean()
            self.lat_centr = self.data[["Del_lat", "Res_lat"]].mean().mean()
        # Создаем коллектор для данных
        self.datacollector = SelectiveDataCollector(
            # model_reporters={"Waiting": proportion_of_waiting_orders},
            agenttype_reporters={
                Order: {
                    "Created": "created_at",
                    "Delivered_at": "delivered_at",
                    "Deadline": "deadline",
                    "Res_loc": "restaurant_location",
                    "Del_loc": "customer_location",
                    "Picked_up": "picked_up",
                    "Delivered": "delivered",
                    "multi_order_index": "multi_order_index",
                    "deliver_id": "courier_id",
                    "Res_id": "restaurant_id",
                    "Del_id": "customer_id",
                    "ready_for_pickup": "ready_for_pickup",
                    "closest_courier_id": "closest_courier_id",
                },
                Courier: {
                    "distance": "distance_covered",
                    "load": "busy",
                    "position": "pos",
                    "serving_order": "picked_order_id",
                    "time_to_deliver": "time_to_deliver",
                    "num_orders": "num_orders",
                    "waiting_in_restaurant": "waiting_in_restaurant",
                    "speed": "speed",
                    "stamina": "stamina",
                    "earnings": "earnings",
                },
                Customer: {
                    "delay_count": "delay_count",
                    "total_delay_time": "total_delay_time",
                    "order_id": "order_id",
                    "last_delay": "last_delay",
                    "customer_id": "customer_id",
                },
                Restaurant: {
                    "current_cooking_orders": "current_cooking_orders",
                    "preparation_capacity": "preparation_capacity",
                    "restaurant_id": "restaurant_id",
                },
            },
        )

        # Создаем курьеров
        for i, courier in enumerate(self.couriers):
            a = Courier(self, self.speed[i], tuple(courier))
            # Рандомное первоначальное положение
            if not self.no_map:
                self.space.place_agent(a, np.array(courier))

        # Создаем все рестораны
        for _, res in (
            self.data[["Res_id", "Res_lat", "Res_long"]].drop_duplicates().iterrows()
        ):
            a = Restaurant(self, res["Res_id"])
            # Рандомное первоначальное положение
            if not self.no_map:
                self.space.place_agent(a, [res["Res_lat"], res["Res_long"]])

    # def simulation(self):
    #     data_order = self.datacollector.get_agenttype_vars_dataframe_last_step_last_step(agent_type=Order)
    #     while (
    #         data_order["Picked_up"].isna().values[0]
    #         or not data_order.loc[self.steps][["Picked_up", "Delivered"]].all().all()
    #         or self.data.shape[0] != data_order.loc[self.steps].shape[0]
    #     ):
    #         self.step()
    #         data_order = self.datacollector.get_agenttype_vars_dataframe_last_step_last_step(
    #             agent_type=Order
    #         )

    def simulation(self, max_iter=2000, w1=0.5, w2=0.5):
        self.orders_entered_system = 0
        print("data shape", self.data.shape[0])
        while (self.data.shape[0] != self.served_orders) and (self.steps <= max_iter):
            if self.steps % 100 == 0:
                number_serving_orders = len(
                    [
                        i
                        for i in self.agents
                        if isinstance(i, Order) and i.courier_id != 0
                    ]
                )
                print(
                    "Step:",
                    self.steps,
                    "Served orders:",
                    self.served_orders,
                    "Orders in work:",
                    number_serving_orders,
                    "Orders entered system",
                    self.orders_entered_system,
                )

            self.step()

        print("Target function val:", self.target_func(w1, w2))

    def target_func(self, w1=0.5, w2=0.5):
        # tardiness
        data_order = self.datacollector.get_agenttype_vars_dataframe(
            agent_type=Order
        ).reset_index()

        data_delivered = data_order[data_order["Delivered"] == True]
        result = (
            data_delivered["Delivered_at"]
            - data_delivered["Created"]
            - data_delivered["Deadline"]
        ).abs()
        tardiness_val = float(result[result > 0].sum() / (15 * result.shape[0]))

        # distance

        data_courier = self.datacollector.get_agenttype_vars_dataframe(
            agent_type=Courier
        ).reset_index()

        distance = (
            data_courier.groupby("AgentID")["distance"].max().mean() / self.max_distance
        )

        return w1 * tardiness_val + w2 * distance

    def step(self):
        """
        Метод для исполнения шага модели
        """
        # Создаем все заказы и располагаем их на карте
        orders = self.data[self.data["Time"] == self.steps]
        if orders.shape[0] > 0:
            for _, order in orders.iterrows():
                self.orders_entered_system += 1
                np.random.seed(self.random_state)
                # preparation_time = 0
                preparation_time = np.random.randint(15, 30)
                # Создаем заказ

                order_agent = Order(
                    self,
                    (order["Res_lat"], order["Res_long"]),
                    order["Res_id"],
                    (order["Del_lat"], order["Del_long"]),
                    order["Del_id"],
                    deadline=order["Deadline"],
                    created_at=order["Time"],
                    preparation_time=preparation_time,  # Pass preparation time to the order
                )
                # Создаем ресторан и заказчика

                if not self.no_map:
                    customer = Customer(self, order_agent.unique_id, order["Del_id"])
                    # Ставим их на карту
                    self.space.place_agent(
                        customer, (order["Del_lat"], order["Del_long"])
                    )
                for restaurant_agent in self.agents:
                    if (
                        isinstance(restaurant_agent, Restaurant)
                        and restaurant_agent.restaurant_id == order_agent.restaurant_id
                    ):
                        restaurant_agent.preparation_queue.append(order_agent)

        # Update Restaurant Status (preparation times)
        self.agents_by_type[Restaurant].shuffle_do("step")
        all_agents = copy(self.agents)
        for agent in all_agents:
            if isinstance(agent, Order):
                if agent.courier_id == 0:
                    agent.closest_courier_id = self.find_closest_courier(
                        agent.restaurant_location[0],
                        agent.restaurant_location[1],
                        only_params=True,
                    ).unique_id
                else:
                    agent.closest_courier_id = agent.courier_id
        self.datacollector.collect(self)
        # Назначаем свободных курьеров на заказы
        self.courier_assingment()
        # Собираем данные о системе

        # Заставляем курьеров выполнить шаг (выбор / доставку заказа )
        self.agents_by_type[Courier].shuffle_do("step")

        self.datacollector.collect(self)

        all_agents = copy(self.agents)
        # future_use_of_restaurant = self.data.loc[
        #     self.data["Time"] >= self.steps, ["Res_id"]
        # ].drop_duplicates()
        for agent in all_agents:
            agent.changed = False
            if isinstance(agent, Order) and agent.delivered == True:
                agent.remove()
                self.served_orders += 1
            # if (
            #     isinstance(agent, Restaurant)
            #     and len(agent.preparation_queue) == 0
            #     and len(agent.preparing_orders) == 0
            #     and agent.restaurant_id in future_use_of_restaurant
            # ):
            #     agent.remove()
            # if isinstance(agent, Customer) and agent.order_delivered == True:
            #     agent.remove()
            if (
                isinstance(agent, Courier)
                and agent.distance_covered > self.max_distance
                and len(agent.order_list) == 0
            ):
                agent.remove()

    def find_closest_courier(self, lat, long, number_orders=1, only_params=False):
        best_courier = None
        best_rank = float("inf")

        agents_data = []

        for agent in self.agents:
            if isinstance(agent, Courier) and ((len(agent.path) == 0) or only_params):
                # Compute distance
                distance = time_moving(
                    agent.pos[0], agent.pos[1], lat, long, agent.speed, agent.stamina
                )

                # Compute earning
                earning = (
                    200 * number_orders
                    + distance * 10
                    + sum(
                        1
                        for i in self.agents
                        if (
                            (isinstance(i, Courier) and len(i.order_list) != 0)
                            or (isinstance(i, Order) and i.courier_id == 0)
                        )
                    )
                    * 20
                    + agent.earnings
                )

                agents_data.append((agent, distance, earning))

        if not agents_data:
            return None

        # Rank agents by distance and earnings using `enumerate`
        agents_data.sort(key=lambda x: x[1])  # Sort by distance
        rank_distance = {
            agent.unique_id: rank
            for rank, (agent, _, _) in enumerate(agents_data, start=1)
        }

        agents_data.sort(key=lambda x: x[2])  # Sort by earnings
        rank_earnings = {
            agent.unique_id: rank
            for rank, (agent, _, _) in enumerate(agents_data, start=1)
        }

        # Find the courier with the lowest combined rank
        for agent, _, earn in agents_data:
            rank = rank_distance[agent.unique_id] + rank_earnings[agent.unique_id]
            if rank < best_rank:
                best_rank = rank
                best_courier = agent
                best_earn = earn
        if not only_params:
            best_courier.earnings = best_earn
        return best_courier

    def courier_assingment(self):
        all_orders = self.datacollector.get_agenttype_vars_dataframe_last_step(
            agent_type=Order,
        )
        free_couriers = len(
            [i for i in self.agents if isinstance(i, Courier) and i.busy == False]
        )
        # print(all_orders)
        # print("all_orders", all_orders.shape)
        if all_orders.shape[0] > 0:
            # print("passed" + "-" * 10)

            current_order_stack = all_orders
            have_orders = (
                current_order_stack[(current_order_stack["deliver_id"] == 0)].shape[0]
                > 0
            )
            if have_orders and free_couriers > 0:
                # Курьеры

                orders_to_be_delivered = self.prepare_dataset()
                orders_to_be_delivered = orders_to_be_delivered[
                    orders_to_be_delivered["deliver_id"] == 0
                ]
                # print(orders_to_be_delivered)

                # print(orders_to_be_delivered.columns)

                data_time = orders_to_be_delivered
                if self.priority == "Deadline":
                    data_time = self.deadline(orders_to_be_delivered)
                elif self.priority == "reserve":
                    data_time = self.reserve(orders_to_be_delivered)
                elif self.priority == "fastest":
                    data_time = self.fastest(orders_to_be_delivered)
                elif self.priority == "longest":
                    data_time = self.longest(orders_to_be_delivered)
                elif self.priority == "ML_fastest":
                    data_time = self.ml_fastest(orders_to_be_delivered)
                elif self.priority == "ML_longest":
                    data_time = self.ml_longest(orders_to_be_delivered)
                if self.aggregation is None:
                    self.define_route(data_time)

                # Агрегация
                else:
                    data_to_cluster = data_time
                    position_columns = [
                        "Res_loc_lat",
                        "Res_loc_long",
                        "Del_loc_lat",
                        "Del_loc_long",
                    ]

                    if data_to_cluster.shape[0] > 1:
                        if self.aggregation == "DBSCAN":
                            clust = DBSCAN(min_samples=1, eps=self.eps)
                            agg = clust.fit_predict(data_to_cluster[position_columns])
                        elif self.aggregation == "KMeans":
                            clust = KMeans(
                                # n_clusters=int(
                                #     0.1 * data_to_cluster.shape[0] // 2
                                #     + 0.1 * data_to_cluster.shape[0] % 2
                                #     + 0.9 * min(free_couriers, data_to_cluster.shape[0])
                                # ),
                                n_clusters=int(data_to_cluster.shape[0] // 1.25)
                                + int(round(data_to_cluster.shape[0] % 1.25, 0)),
                                random_state=self.random_state,
                            )
                            agg = clust.fit_predict(data_to_cluster[position_columns])
                        elif self.aggregation == "Agglo":
                            clust = AgglomerativeClustering(
                                distance_threshold=self.eps, n_clusters=None
                            )
                            agg = clust.fit_predict(data_to_cluster[position_columns])
                        data_to_cluster["agg"] = agg
                        if self.priority == "cluster_reserve":
                            data_to_cluster = self.reserve(data_to_cluster)
                        if self.priority == "cluster_longest":
                            data_to_cluster = self.longest(data_to_cluster)
                        if self.priority == "cluster_fastest":
                            data_to_cluster = self.fastest(data_to_cluster)

                        if self.priority in [
                            "cluster_reserve",
                            "cluster_longest",
                            "cluster_fastest",
                        ]:
                            dict_agg = dict(
                                zip(
                                    data_to_cluster["agg"].unique(),
                                    [0] * data_to_cluster["agg"].unique().shape[0],
                                )
                            )
                        if self.priority in [
                            "cluster_reserve",
                            "cluster_longest",
                            "cluster_fastest",
                        ]:
                            for cluster in data_to_cluster["agg"].unique():
                                data_in_cluster = data_to_cluster[
                                    data_to_cluster["agg"] == cluster
                                ]
                                path_sum = self.get_route(
                                    data_in_cluster, only_dist=True, courier_pos=None
                                )
                                if self.priority == "cluster_reserve":
                                    if path_sum == 0 or path_sum is None:
                                        dict_agg[cluster] = (
                                            data_in_cluster.iloc[-1]["Deadline"]
                                            / 0.0001
                                        )
                                    else:
                                        dict_agg[cluster] = (
                                            data_in_cluster.iloc[-1]["Deadline"]
                                            / path_sum
                                        )
                                else:
                                    if path_sum is not None:
                                        dict_agg[cluster] = path_sum
                                    else:
                                        dict_agg[cluster] = 100_000

                                if self.priority in [
                                    "cluster_reserve",
                                    "cluster_fastest",
                                ]:
                                    cluster_order = list(
                                        dict(
                                            sorted(
                                                dict_agg.items(),
                                                key=lambda x: x[1],
                                                reverse=False,
                                            )
                                        ).keys()
                                    )
                                else:
                                    cluster_order = list(
                                        dict(
                                            sorted(
                                                dict_agg.items(),
                                                key=lambda x: x[1],
                                                reverse=True,
                                            )
                                        ).keys()
                                    )

                        else:
                            cluster_order = data_to_cluster["agg"].unique()

                        for cluster_label in cluster_order:
                            agg_order = data_to_cluster[
                                data_to_cluster["agg"] == cluster_label
                            ].copy()
                            if cluster_label == -1:
                                for i, order in agg_order.iterrows():
                                    self.define_route(order)
                            else:

                                self.define_route(
                                    agg_order.head(2),
                                    agg_order.head(2)["AgentID"].values[0],
                                    to_clust=True,
                                )

                    elif data_to_cluster.shape[0] == 1:
                        self.define_route(
                            data_to_cluster, data_to_cluster["AgentID"].values[0]
                        )

    def train_ml_model(self, historical_data: pd.DataFrame):
        """
        Train a machine learning model to predict the time to deliver an order.

        Parameters:
        historical_data (pd.DataFrame): Historical data containing features and target variable.
        """
        # Ensure the historical data contains the necessary columns
        self.required_columns = [
            "Step",
            "Created",
            "Deadline",
            "Res_id",
            "Del_id",
            "Res_loc_lat",
            "Res_loc_long",
            "Del_loc_lat",
            "Del_loc_long",
            "distance",
            "load",
            "time_to_deliver",
            "waiting_in_restaurant",
            "delay_count",
            "total_delay_time",
            "current_cooking_orders",
            "preparation_capacity",
            "speed",
            "stamina",
            "earnings",
        ]
        if not all(
            col in historical_data.columns
            for col in self.required_columns + ["time_to_serve"]
        ):
            raise ValueError(
                f"Historical data must contain the following columns: {self.required_columns}"
            )

        # Prepare features and target variable
        X = historical_data[self.required_columns]
        y = historical_data["time_to_serve"]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )

        # Initialize and train the model
        self.pre_trained_model_predict_p_i.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.pre_trained_model_predict_p_i.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Model trained with Mean Squared Error: {mse}")

    def predict_delivery_time(self, order_data: pd.DataFrame):
        """
        Predict the time to deliver an order using the trained ML model.

        Parameters:
        order_data (pd.DataFrame): Data containing the features of the order.

        Returns:
        float: Predicted time to deliver the order.
        """
        self.required_columns = [
            "Step",
            "AgentID",
            "Created",
            "Deadline",
            "Res_id",
            "Del_id",
            "Res_loc_lat",
            "Res_loc_long",
            "Del_loc_lat",
            "Del_loc_long",
            "distance",
            "load",
            "time_to_deliver",
            "waiting_in_restaurant",
            "delay_count",
            "last_delay",
            "total_delay_time",
            "current_cooking_orders",
            "preparation_capacity",
            "speed",
            "stamina",
            "earnings",
        ]

        # Ensure the order data contains the necessary columns
        if not all(col in order_data.columns for col in self.required_columns):
            raise ValueError(
                f"Order data must contain the following columns: {self.required_columns}"
            )

        # Predict the delivery time
        predicted_time = self.pre_trained_model_predict_p_i.predict(
            order_data[self.required_columns]
        )
        return predicted_time[0]

    def define_route(self, orders, multi_index=0, to_clust=False):
        closest_courier = None
        if to_clust:
            closest_courier = self.find_closest_courier(
                orders["Res_loc_lat"].mean(),
                orders["Res_loc_long"].mean(),
                orders.shape[0],
            )
            if closest_courier is not None:
                visiting_order, idx = self.get_route(
                    orders, courier_pos=closest_courier.pos
                )
                if visiting_order is not None:
                    for i, destination in enumerate(visiting_order[1:-1]):
                        name_destination = idx[destination]
                        if name_destination[:3] == "Res":
                            for restaurant, location_lat, location_long in orders.loc[
                                orders["Res_id"] == int(name_destination[4:]),
                                ["AgentID", "Res_loc_lat", "Res_loc_long"],
                            ].values:
                                closest_courier.path.extend(
                                    [
                                        [
                                            f"Restaurant_{int(restaurant)}",
                                            (location_lat, location_long),
                                        ]
                                    ]
                                )
                        else:
                            for customer, location_lat, location_long in orders.loc[
                                orders["Del_id"] == int(name_destination[4:]),
                                ["AgentID", "Del_loc_lat", "Del_loc_long"],
                            ].values:
                                closest_courier.path.extend(
                                    [
                                        [
                                            f"Customer_{int(customer)}",
                                            (location_lat, location_long),
                                        ]
                                    ]
                                )

        for _, order in orders.iterrows():
            if self.aggregation is None or not to_clust:
                closest_courier = self.find_closest_courier(
                    order["Res_loc_lat"], order["Res_loc_long"], 1
                )
            closest_order_id = order["AgentID"]
            for order_agent in self.agents:
                if (
                    isinstance(order_agent, Order)
                    and order_agent.unique_id == closest_order_id
                ):
                    closest_order = order_agent
                    break
            if closest_courier is not None:
                closest_order.courier_id = closest_courier.unique_id
                closest_courier.busy = True
                closest_courier.order_list.append(closest_order)
                if self.priority in ("ML_fastest", "ML_longest"):
                    closest_order.deadline = order["time_to_serve"]
                if not to_clust or visiting_order is None:
                    closest_courier.path.extend(
                        [
                            [
                                f"Restaurant_{order['AgentID']}",
                                closest_order.restaurant_location,
                            ],
                            [
                                f"Customer_{order['AgentID']}",
                                closest_order.customer_location,
                            ],
                        ]
                    )
                closest_order.multi_order_index = multi_index

            else:
                break

    # @staticmethod
    # def data_generation(
    #     num_orders: int = 10,
    #     max_time: float = 10,
    #     num_rest: int = 5,
    #     random_state: int = 42,
    # ):
    #     np.random.seed(random_state)
    #     rest = np.random.normal(5, 0.05, (num_rest, 2))
    #     rest_idx = np.random.randint(0, num_rest, size=num_orders)
    #     rest_matr = rest[rest_idx]

    #     customer_loc = np.random.normal(5, 0.13, (num_orders * 2, 2))
    #     customer_idx = np.random.randint(0, num_orders * 2, size=num_orders)
    #     customer_loc = customer_loc[customer_idx]

    #     test_matr = [
    #         [
    #             np.random.randint(1, max_time),
    #             rest_idx[i],
    #             customer_idx[i],
    #             rest_matr[i, 0],
    #             rest_matr[i, 1],
    #             customer_loc[i, 0],
    #             customer_loc[i, 1],
    #             np.random.randint(30, 50),
    #         ]
    #         for i in range(num_orders)
    #     ]
    #     test_names = [
    #         "Time",
    #         "Res_id",
    #         "Del_id",
    #         "Res_lat",
    #         "Res_long",
    #         "Del_lat",
    #         "Del_long",
    #         "Deadline",
    #     ]

    #     data_test = pd.DataFrame(test_matr, columns=test_names)
    #     data_test = data_test.sort_values("Time").reset_index(drop=True).reset_index()
    #     return data_test

    @staticmethod
    def data_generation(
        num_orders: int = 1000,
        max_time: float = 1440,
        num_restaurants: int = 20,
        num_customers: int = 50,
        peak_times: list = [720, 1080],
        peak_intensity: float = 3.0,
        scale: float = 1,
        city_center: tuple = (55.751244, 37.618423),
        random_state: int = 42,
        res_scale: list = [0.01, 0.02],
        customer_scale: list = [0.03, 0.04],
        portion_customers_normal: float = 0.5,
    ) -> pd.DataFrame:
        """
        Generates synthetic delivery orders with temporal peaks and spatial clustering

        Parameters:
        num_orders: Total number of orders
        max_time: Simulation duration in minutes
        num_restaurants: Number of unique restaurants
        num_customers: Number of available delivery personnel
        peak_times: List of peak time centers (in minutes)
        peak_intensity: Intensity of peak clustering (higher = more concentrated)
        city_center: Central coordinates for spatial distribution
        """
        np.random.seed(random_state)

        # Calculate base orders per peak
        base_orders = num_orders // len(peak_times)
        remaining = num_orders % len(peak_times)

        # Generate peak orders
        peak_orders = []
        for peak in peak_times:
            # Generate base orders for this peak
            peak_dist = skewnorm.rvs(
                peak_intensity, loc=peak, scale=scale, size=base_orders
            )
            peak_orders.append(np.clip(peak_dist, 0, max_time))

        # Generate remaining orders uniformly
        if remaining > 0:
            uniform_orders = np.random.uniform(1, max_time, remaining)
            # Distribute remaining orders across peaks
            for i in range(remaining):
                peak_orders[i % len(peak_times)] = np.append(
                    peak_orders[i % len(peak_times)], uniform_orders[i]
                )

        # Flatten and shuffle orders
        order_times = np.concatenate(peak_orders)
        order_times = np.random.permutation(order_times).astype(int)

        # Generate locations
        rest_locs = np.random.normal(
            loc=city_center, scale=res_scale, size=(num_restaurants, 2)
        )
        rest_ids = np.random.randint(0, num_restaurants, num_orders)

        delivery_locs = np.concatenate(
            [
                np.random.normal(
                    loc=city_center,
                    scale=customer_scale,
                    size=(int(num_customers * portion_customers_normal), 2),
                ),
                np.random.uniform(
                    low=[city_center[0] - 0.1, city_center[1] - 0.2],
                    high=[city_center[0] + 0.1, city_center[1] + 0.2],
                    size=(
                        num_customers - int(num_customers * portion_customers_normal),
                        2,
                    ),
                ),
            ]
        )
        deliver_ids = np.random.randint(0, num_customers, num_orders)
        # print(len(order_times))
        # print(len(delivery_locs[:, 0]))
        # Create dataframe
        df = (
            pd.DataFrame(
                {
                    "Time": order_times,
                    "Res_id": rest_ids,
                    "Del_id": deliver_ids,
                    "Res_lat": rest_locs[rest_ids, 0],
                    "Res_long": rest_locs[rest_ids, 1],
                    "Del_lat": delivery_locs[deliver_ids, 0],
                    "Del_long": delivery_locs[deliver_ids, 1],
                    "Deadline": np.clip(
                        np.abs(np.random.normal(45, 15, num_orders)), 30, 120
                    ),
                }
            )
            .sort_values("Time")
            .reset_index(drop=True)
            .reset_index()
            # .rename(columns={"index": "Order_ID"})
        )
        df.loc[df["Time"] == 0, "Time"] = 1
        # Adjust deadlines during peaks
        df["Deadline"] *= np.where(
            df["Time"].between(peak_times[0] - 60, peak_times[0] + 60)
            | df["Time"].between(peak_times[1] - 60, peak_times[1] + 60),
            1.2,
            0.9,
        )

        return df

    @staticmethod
    def courier_generation(
        data_test: pd.DataFrame,
        num_couriers: int,
        left_quantile: float = 0.15,
        right_quantile: float = 0.85,
        random_state: int = 42,
    ):
        np.random.seed(random_state)
        couriers = []
        for i in range(num_couriers):
            couriers.append(
                (
                    np.random.uniform(
                        data_test["Res_lat"].quantile(left_quantile),
                        data_test["Res_lat"].quantile(right_quantile),
                    ),
                    np.random.uniform(
                        data_test["Res_long"].quantile(left_quantile),
                        data_test["Res_long"].quantile(right_quantile),
                    ),
                )
            )
        return couriers

    def random_position(self):
        """
        Метод для генерации случайного местоположения на карте
        """
        return (
            self.random.randint(0, self.space.width - 1),
            self.random.randint(0, self.space.height - 1),
        )

    @staticmethod
    def deadline(data_time):
        return data_time.sort_values("Deadline")

    @staticmethod
    def reserve(data_time):
        position_columns = [
            "Res_loc_lat",
            "Res_loc_long",
            "Del_loc_lat",
            "Del_loc_long",
            "speed",
            "stamina",
        ]
        data_time = data_time.copy()
        data_time["reserve"] = data_time["Deadline"] / data_time[
            position_columns
        ].apply(
            lambda x: time_moving(
                x.Res_loc_lat,
                x.Res_loc_long,
                x.Del_loc_lat,
                x.Del_loc_long,
                x.speed,
                x.stamina,
            ),
            axis=1,
        )
        return data_time.sort_values("reserve", ascending=False)

    @staticmethod
    def fastest(data_time):
        position_columns = [
            "Res_loc_lat",
            "Res_loc_long",
            "Del_loc_lat",
            "Del_loc_long",
            "speed",
            "stamina",
        ]
        data_time = data_time.copy()
        data_time["time_to_serve"] = data_time[position_columns].apply(
            lambda x: time_moving(
                x.Res_loc_lat,
                x.Res_loc_long,
                x.Del_loc_lat,
                x.Del_loc_long,
                x.speed,
                x.stamina,
            ),
            axis=1,
        )
        return data_time.sort_values("time_to_serve")

    @staticmethod
    def longest(data_time):
        position_columns = [
            "Res_loc_lat",
            "Res_loc_long",
            "Del_loc_lat",
            "Del_loc_long",
            "speed",
            "stamina",
        ]
        data_time = data_time.copy()
        data_time["time_to_serve"] = data_time[position_columns].apply(
            lambda x: time_moving(
                x.Res_loc_lat,
                x.Res_loc_long,
                x.Del_loc_lat,
                x.Del_loc_long,
                x.speed,
                x.stamina,
            ),
            axis=1,
        )
        return data_time.sort_values("time_to_serve", ascending=False)

    def ml_fastest(self, data_time):

        data_time = data_time.copy()
        data_time["time_to_serve"] = self.predict_delivery_time(data_time)

        return data_time.sort_values("time_to_serve", ascending=True)

    def ml_longest(self, data_time):

        data_time = data_time.copy()
        data_time["time_to_serve"] = self.predict_delivery_time(data_time)

        return data_time.sort_values("time_to_serve", ascending=False)

    def create_data_model(self, orders, courier_pos=None):
        """Stores the data for the problem."""
        if courier_pos is None:
            courier_pos = (orders["Res_loc_lat"].mean(), orders["Res_loc_long"].mean())

        dist = []

        self._idx = (
            ["courier"]
            + ["Res_" + str(int(i)) for i in orders["Res_id"].unique()]
            + ["Del_" + str(int(i)) for i in orders["Del_id"].unique()]
        )

        order = []
        for row in orders[["Res_id", "Del_id"]].values:
            order.append(
                [
                    self._idx.index(f"Res_{int(row[0])}"),
                    self._idx.index(f"Del_{int(row[1])}"),
                ]
            )
        for i in self._idx:
            row = []
            for j in self._idx:
                if (i == "courier") and (j != "courier"):
                    row.append(
                        int(
                            time_moving(
                                courier_pos[0],
                                courier_pos[1],
                                orders.loc[
                                    orders[f"{j[:3]}_id"] == int(j[4:]),
                                    f"{j[:3]}_loc_lat",
                                ].values[0],
                                orders.loc[
                                    orders[f"{j[:3]}_id"] == int(j[4:]),
                                    f"{j[:3]}_loc_long",
                                ].values[0],
                            )
                        )
                    )
                elif (i != "courier") and (j == "courier"):
                    row.append(
                        int(
                            time_moving(
                                orders.loc[
                                    orders[f"{i[:3]}_id"] == int(i[4:]),
                                    f"{i[:3]}_loc_lat",
                                ].values[0],
                                orders.loc[
                                    orders[f"{i[:3]}_id"] == int(i[4:]),
                                    f"{i[:3]}_loc_long",
                                ].values[0],
                                courier_pos[0],
                                courier_pos[1],
                            )
                        )
                    )
                elif (i == "courier") and (j == "courier"):
                    row.append(0)
                else:
                    row.append(
                        int(
                            time_moving(
                                orders.loc[
                                    orders[f"{i[:3]}_id"] == int(i[4:]),
                                    f"{i[:3]}_loc_lat",
                                ].values[0],
                                orders.loc[
                                    orders[f"{i[:3]}_id"] == int(i[4:]),
                                    f"{i[:3]}_loc_long",
                                ].values[0],
                                orders.loc[
                                    orders[f"{j[:3]}_id"] == int(j[4:]),
                                    f"{j[:3]}_loc_lat",
                                ].values[0],
                                orders.loc[
                                    orders[f"{j[:3]}_id"] == int(j[4:]),
                                    f"{j[:3]}_loc_long",
                                ].values[0],
                            )
                        )
                    )
            dist.append(row)

        data = {}
        data["distance_matrix"] = dist
        data["pickups_deliveries"] = order
        data["num_vehicles"] = 1
        data["depot"] = 0
        return data, self._idx

    def get_route(self, orders, only_dist=False, courier_pos=None):
        """Entry point of the program."""
        # Instantiate the data problem.
        data, idx = self.create_data_model(orders, courier_pos)

        # Create the routing index manager.
        manager = pywrapcp.RoutingIndexManager(
            len(data["distance_matrix"]), data["num_vehicles"], data["depot"]
        )

        # Create Routing Model.
        routing = pywrapcp.RoutingModel(manager)

        # Define cost of each arc.
        def distance_callback(from_index, to_index):
            """Returns the manhattan distance between the two nodes."""
            # Convert from routing variable Index to distance matrix NodeIndex.
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data["distance_matrix"][from_node][to_node]

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add Distance constraint.
        dimension_name = "Distance"
        routing.AddDimension(
            transit_callback_index,
            0,  # no slack
            30000,  # vehicle maximum travel distance
            True,  # start cumul to zero
            dimension_name,
        )
        distance_dimension = routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Define Transportation Requests.
        for request in data["pickups_deliveries"]:
            pickup_index = manager.NodeToIndex(request[0])
            delivery_index = manager.NodeToIndex(request[1])
            routing.AddPickupAndDelivery(pickup_index, delivery_index)
            routing.solver().Add(
                routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
            )
            routing.solver().Add(
                distance_dimension.CumulVar(pickup_index)
                <= distance_dimension.CumulVar(delivery_index)
            )

        # Setting first solution heuristic.
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )

        # Solve the problem.
        solution = routing.SolveWithParameters(search_parameters)

        if solution:
            # print(f"Objective: {solution.ObjectiveValue()}")
            total_distance = 0
            for vehicle_id in range(data["num_vehicles"]):
                index = routing.Start(vehicle_id)
                plan_output = []
                route_distance = 0
                while not routing.IsEnd(index):
                    plan_output.append(manager.IndexToNode(index))
                    previous_index = index
                    index = solution.Value(routing.NextVar(index))
                    route_distance += routing.GetArcCostForVehicle(
                        previous_index, index, vehicle_id
                    )
                plan_output.append(manager.IndexToNode(index))
                # print(plan_output)
                total_distance += route_distance
            # print(f"Total Distance of all routes: {total_distance}km")
            if not only_dist:
                return plan_output, idx
            else:
                return route_distance / len(data["distance_matrix"])

        else:
            print("No solution")
            if not only_dist:
                return None, None
            else:
                return None

    def prepare_dataset(self, return_full=False, return_all=False):
        if not return_all:
            all_orders_data = self.datacollector.get_agenttype_vars_dataframe_last_step(
                agent_type=Order,
            ).reset_index()
        else:
            all_orders_data = self.datacollector.get_agenttype_vars_dataframe(
                agent_type=Order,
            ).reset_index()
        all_orders_data["Res_loc_lat"] = all_orders_data["Res_loc"].apply(
            lambda x: x[0]
        )
        all_orders_data["Res_loc_long"] = all_orders_data["Res_loc"].apply(
            lambda x: x[1]
        )

        all_orders_data["Del_loc_lat"] = all_orders_data["Del_loc"].apply(
            lambda x: x[0]
        )
        all_orders_data["Del_loc_long"] = all_orders_data["Del_loc"].apply(
            lambda x: x[1]
        )
        if not return_all:
            all_couriers_data = (
                self.datacollector.get_agenttype_vars_dataframe_last_step(
                    agent_type=Courier,
                )
            )
        else:
            all_couriers_data = self.datacollector.get_agenttype_vars_dataframe(
                agent_type=Courier,
            )
        all_couriers_data["load"] = all_couriers_data["load"].astype(int)

        if (self.priority is not None and "ml" in self.priority.lower()) or (
            return_full
        ):
            if not return_all:
                all_customers_data = (
                    self.datacollector.get_agenttype_vars_dataframe_last_step(
                        agent_type=Customer,
                    )
                )
            else:
                all_customers_data = self.datacollector.get_agenttype_vars_dataframe(
                    agent_type=Customer,
                )
            if not return_all:
                all_restaurants_data = (
                    self.datacollector.get_agenttype_vars_dataframe_last_step(
                        agent_type=Restaurant,
                    )
                )
            else:
                all_restaurants_data = self.datacollector.get_agenttype_vars_dataframe(
                    agent_type=Restaurant,
                )

            # Merge all data into a single DataFrame
            historical_data = (
                all_orders_data.merge(
                    all_couriers_data.reset_index(),
                    how="left",
                    left_on=["Step", "closest_courier_id"],
                    right_on=["Step", "AgentID"],
                    suffixes=["_order", "_courier"],
                )
                .merge(
                    all_customers_data.reset_index(),
                    how="left",
                    left_on=["Step", "AgentID_order"],
                    right_on=["Step", "order_id"],
                    suffixes=["", "_customer"],
                )
                .merge(
                    all_restaurants_data.reset_index(),
                    how="left",
                    left_on=["Step", "Res_id"],
                    right_on=["Step", "restaurant_id"],
                    suffixes=["", "_restaurant"],
                )
            )
            if return_all:
                historical_data = historical_data.merge(
                    all_orders_data[all_orders_data["Delivered"] == True],
                    how="left",
                    left_on="AgentID_order",
                    right_on="AgentID",
                    suffixes=["", "_delivered"],
                )

                # Calculate the time it took to deliver each order
                historical_data["time_to_serve"] = (
                    historical_data["Delivered_at_delivered"] - historical_data["Step"]
                )
                historical_data["AgentID"] = historical_data["AgentID_order"]
                # Extract relevant features and target variable
                historical_data = historical_data[
                    # (historical_data["deliver_id"] == 0),
                    [
                        "Step",
                        "AgentID",
                        "Created",
                        "Deadline",
                        "Res_id",
                        "Del_id",
                        "Res_loc_lat",
                        "Res_loc_long",
                        "Del_loc_lat",
                        "Del_loc_long",
                        "distance",
                        "load",
                        "time_to_deliver",
                        "waiting_in_restaurant",
                        "delay_count",
                        "last_delay",
                        "total_delay_time",
                        "current_cooking_orders",
                        "preparation_capacity",
                        "speed",
                        "stamina",
                        "earnings",
                        "deliver_id",
                        "time_to_serve",  # Target variable
                    ]
                ]
            else:
                historical_data["AgentID"] = historical_data["AgentID_order"]
                # Extract relevant features and target variable

                historical_data = historical_data[
                    # (historical_data["deliver_id"] == 0),
                    [
                        "Step",
                        "AgentID",
                        "Created",
                        "Deadline",
                        "Res_id",
                        "Del_id",
                        "Res_loc_lat",
                        "Res_loc_long",
                        "Del_loc_lat",
                        "Del_loc_long",
                        "distance",
                        "load",
                        "time_to_deliver",
                        "waiting_in_restaurant",
                        "delay_count",
                        "last_delay",
                        "total_delay_time",
                        "current_cooking_orders",
                        "preparation_capacity",
                        "speed",
                        "stamina",
                        "earnings",
                        "deliver_id",
                        # "time_to_serve",  # Target variable
                    ]
                ]

            # Drop rows with missing values (e.g., orders that were not delivered)
            # historical_data = historical_data.dropna()

        else:
            historical_data = all_orders_data.merge(
                all_couriers_data.reset_index(),
                how="left",
                left_on=["Step", "closest_courier_id"],
                right_on=["Step", "AgentID"],
                suffixes=["_order", "_courier"],
            )
            historical_data = historical_data[(historical_data["deliver_id"] == 0)]
            historical_data["AgentID"] = historical_data["AgentID_order"]
        return historical_data
