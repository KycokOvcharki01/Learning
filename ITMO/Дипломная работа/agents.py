from geopy.distance import geodesic
import numpy as np
import contextlib
import itertools
import types
import warnings
from copy import deepcopy
from functools import partial

with contextlib.suppress(ImportError):
    import pandas as pd

# Для гистограммы


import mesa


class SelectiveDataCollector(mesa.DataCollector):
    """
    A variant of Mesa's DataCollector that only collects data from agents
    whose `changed` attribute is True.
    """

    def __init__(
        self,
        model_reporters=None,
        agent_reporters=None,
        agenttype_reporters=None,
        tables=None,
    ):
        super().__init__(model_reporters, agent_reporters, agenttype_reporters, tables)

    def get_agenttype_vars_dataframe_last_step(self, agent_type):
        """Create a pandas DataFrame from the agent-type variables for a specific agent type.

        The DataFrame has one column for each variable, with two additional
        columns for tick and agent_id.

        Args:
            agent_type: The type of agent to get the data for.
        """
        # Check if self.agenttype_reporters dictionary is empty for this agent type, if so return empty DataFrame
        if agent_type not in self.agenttype_reporters:
            warnings.warn(
                f"No agent-type reporters have been defined for {agent_type} in the DataCollector, returning empty DataFrame.",
                UserWarning,
                stacklevel=2,
            )
            return pd.DataFrame()
        model_step = max(self._agenttype_records.keys())
        # print("model_step", model_step)
        # print("1", self._agenttype_records.values())
        # print("2", self._agenttype_records[model_step].values())
        # print("3", self._agenttype_records)
        # print("4", self._agenttype_records[model_step][agent_type])

        # all_records = self._agenttype_records[model_step][agent_type]
        # print("all_records", all_records)

        all_records = self._agenttype_records[model_step][agent_type]

        rep_names = list(self.agenttype_reporters[agent_type])

        df = pd.DataFrame.from_records(
            data=all_records,
            columns=["Step", "AgentID", *rep_names],
        )
        return df


class Restaurant(mesa.Agent):
    def __init__(self, model, restaurant_id=0, preparation_capacity=1):
        super().__init__(model)
        self.restaurant_id = restaurant_id
        self.preparing_orders = []  # Orders currently being prepared
        self.preparation_queue = []  # Queue of orders waiting for preparation
        self.preparation_capacity = (
            preparation_capacity  # Max orders being cooked at once
        )
        self.changed = True

    @property
    def current_cooking_orders(self):
        """Return the number of orders currently being prepared."""
        return len(self.preparing_orders)

    def step(self):
        initial_length = len(self.preparing_orders)  # save initial state
        # Process preparing orders
        # order_list = deepcopy(self.preparing_orders)
        for order in self.preparing_orders:
            if (
                self.model.steps - order.preparation_start_time
                >= order.preparation_time
            ):
                order.finish_preparation()
                self.preparing_orders.remove(order)  # Remove order from preparing list
        # self.preparing_orders = order_list
        # If has space for new orders, start preparation
        while (
            len(self.preparing_orders) < self.preparation_capacity
            and len(self.preparation_queue) > 0
        ):
            order_to_prepare = self.preparation_queue.pop(0)
            order_to_prepare.start_preparation(
                self.model.steps
            )  # Set preparation start time
            self.preparing_orders.append(order_to_prepare)

        if len(self.preparing_orders) != initial_length:  # Check if it was change
            self.changed = True
        else:
            self.changed = False


class Customer(mesa.Agent):
    def __init__(self, model, order_id, customer_id=0):
        super().__init__(model)
        self.order_id = order_id
        self.delay_count = 0  # Number of delayed deliveries
        self.total_delay_time = 0  # Total delay time
        self.last_delay = 0
        self.customer_id = customer_id
        self.order_delivered = False
        self.changed = True

    def register_delay(self, delay_time):
        """Increase delay count and add delay time."""
        if self.delay_count != 0:
            self.changed = True
        self.delay_count += 1
        self.last_delay = 1
        self.total_delay_time += delay_time


class Order(mesa.Agent):
    def __init__(
        self,
        model,
        restaurant_location,
        restaurant_id,
        customer_location,
        customer_id,
        deadline=0,
        created_at=0,
        multi_order_index=False,
        preparation_time=0,
    ):
        super().__init__(model)
        self.restaurant_id = restaurant_id
        self.customer_id = customer_id
        self.restaurant_location = restaurant_location
        self.customer_location = customer_location
        self.deadline = deadline
        self.picked_up = False
        self.delivered = False
        self.courier_id = 0
        self.multi_order_index = multi_order_index
        self.created_at = created_at
        self.delivered_at = None
        self.preparation_time = preparation_time  # Store preparation time
        self.preparation_start_time = None  # Start of preparation
        self.is_preparing = False  # Is order being prepared?
        self.ready_for_pickup = False  # Is order ready for pick up?
        self.closest_courier_id = -1
        self.changed = True

    def start_preparation(self, time):
        self.is_preparing = True
        self.preparation_start_time = time
        self.changed = True

    def finish_preparation(self):
        self.is_preparing = False
        self.ready_for_pickup = True
        self.changed = True


class MultiOrder(mesa.Agent):
    def __init__(self, model, restaurant_locations, customer_locations):
        super().__init__(model)
        self.restaurant_location = np.mean(restaurant_locations)
        self.customer_location = np.mean(customer_locations)
        self.picked_up = [False] * len(restaurant_locations)
        self.delivered = [False] * len(customer_locations)
        self.courier_id = 0


def time_moving(
    start_lat,
    start_long,
    finish_lat,
    finish_long,
    speed=1,
    stamina=1,
    random_speed_coef=1,
    fast=True,
):
    if fast:
        return ((start_lat - finish_lat) ** 2 + (start_long - finish_long) ** 2) / (
            speed * stamina * random_speed_coef
        )
    else:
        return geodesic((start_lat, start_long), (finish_lat, finish_long)).km / (
            speed * stamina * random_speed_coef
        )


# Класс с доставщиком
class Courier(mesa.Agent):

    def __init__(self, model, speed, pos):
        super().__init__(model)
        if self.model.no_map:
            self.pos = pos
        # Флаг занятости. Если занят, то не ищет заказ
        self.busy = False
        # Переменная для хранения заказа, который доставляет курьер
        self.order_list = []
        self.num_orders = 0
        # Пройденная дистация курьером
        self.distance_covered = 0
        # ID заказа, доставку которого выполняет курьер
        self.picked_order_id = -1
        # скорость
        self.speed = speed
        # оставшееся время доставки
        self.time_to_deliver = -1
        # Порядок посещения мест
        self.path = []
        self.deliver_stage = 0
        self.on_location = False
        self.waiting_in_restaurant = False
        self.stamina = 1
        self.earnings = 0
        self.changed = True
        self.last_restaurant = None

    def step(self):
        """
        Метод, который определяет действия курьера в каждый момент времени.
        Состоит из двух основных частей:
            - Поиск заказа
            - Выполнение доставки
        """
        initial_stamina = self.stamina
        initial_position = self.pos
        initial_busy = self.busy
        initial_deliver_stage = self.deliver_stage
        initial_waiting_in_restaurant = self.waiting_in_restaurant

        # Ищем заказ
        # if not self.busy:
        #     if self.model.verbose:
        #         print(
        #             f"Step {self.model.steps}: Courier {self.unique_id} is looking for order",
        #         )
        #     # Ищем самый близкицй заказ к курьеру
        #     self.current_order = self.find_closest_neighbor(Order)
        if len(self.order_list) == 0:
            self.stamina = min(1, self.stamina + 0.01)
            if self.model.free_courier_movement == "Center" and (
                time_moving(
                    self.pos[0],
                    self.pos[1],
                    self.model.long_centr,
                    self.model.lat_centr,
                )
                >= 0.005
            ):
                self.move_towards((self.model.long_centr, self.model.lat_centr))
            if (
                self.model.free_courier_movement == "Last"
                and self.last_restaurant is not None
                and (
                    time_moving(
                        self.pos[0],
                        self.pos[1],
                        self.last_restaurant[0],
                        self.last_restaurant[1],
                    )
                    >= 0.005
                )
            ):
                self.move_towards((self.last_restaurant[0], self.last_restaurant[1]))
        # Если есть заказ, который нужно доставить, то доставляем
        if len(self.order_list) > 0:
            # Записываем нужные статусы курьеру и заказу
            # Это в том числе нужно для предотвращения выполнения одного заказа двумя курьерами
            # print("self.order_list", self.order_list)
            if self.model.debug:
                print("courier_id", self.unique_id)
                print("self.path", self.path)
                print("self.path value", self.path[self.deliver_stage])
                print("self.deliver_stage", self.deliver_stage)
                print("self.pos", self.pos)
                print("self.stamina", self.stamina)

            target = self.path[self.deliver_stage][1]
            self.set_current_order()
            if self.model.debug:
                print(
                    "self.current_order.is_preparing", self.current_order.is_preparing
                )
                print(
                    "self.current_order.ready_for_pickup",
                    self.current_order.ready_for_pickup,
                )
            self.num_orders = len(self.order_list)

            self.picked_order_id = self.current_order.unique_id
            if self.model.verbose:
                print(
                    f"Step {self.model.steps}: Courier {self.unique_id} is working on {self.picked_order_id}"
                )

            # Выбираем цель для курьера
            # Сначала надо посетить ресторан
            # Потом заказчика
            # if not self.current_order.picked_up:
            #     target = self.current_order.restaurant_location
            # elif not self.current_order.delivered:
            #     target = self.current_order.customer_location

            # Двигаемся к нужной точке
            self.move_towards(target)

            # Если мы доставили заказ, то убираем заказчика и ресторан с карты
            # Курьер становится свободным
            if self.on_location:

                self.on_location = False
                # Ищем ресторан и заказчика с нужным ID, чтобы убрать с карты
                if not self.model.no_map:
                    for agent in self.model.agents:
                        if (
                            isinstance(agent, Customer)
                        ) and agent.order_id == self.current_order.unique_id:
                            # Функция для удаления агента с карты
                            agent.order_delivered = True
                            self.model.space.remove_agent(agent)
                            break

                # Обновляем статусы агента
                if self.deliver_stage >= len(self.path):
                    self.order_list = []
                    self.busy = False
                    self.picked_order_id = -1
                    self.path = []
                    self.deliver_stage = 0

        if (
            self.stamina != initial_stamina
            or np.all(self.pos != initial_position)
            or self.busy != initial_busy
            or self.deliver_stage != initial_deliver_stage
            or self.waiting_in_restaurant != initial_waiting_in_restaurant
        ):
            self.changed = True
        else:
            self.changed = False

    def move_towards(self, target):
        """
        Метод для передвижения курьера.
        Двигаемся на одну ячейку за один шаг модели
        """
        # Текущая координата
        x, y = self.pos
        # Куда хотим попасть
        tx, ty = target
        # np.random.seed(self.model._seed)
        random_dev = max(min(np.random.normal(1, 0.2), 1.1), 0.01)
        time = time_moving(
            x,
            y,
            tx,
            ty,
            self.speed,
            self.stamina,
            random_dev,
            fast=False,
        )
        self.time_to_deliver = time
        if time <= 1:
            new_pos = (tx, ty)
        else:
            new_pos = (x + (tx - x) / time, y + (ty - y) / time)

        # Обновляем пройденную дистанцию
        self.distance_covered += min(1, time) * self.speed * self.stamina * random_dev

        self.stamina = max(0.01, self.stamina - 0.0001)

        # Обновляем статусы заказа, если оказались в нужной точке
        if (
            (len(self.order_list) > 0)
            and (self.path[self.deliver_stage][1] == new_pos)
            and (self.path[self.deliver_stage][0].split("_")[0] == "Restaurant")
        ):
            if self.current_order.ready_for_pickup == False:
                self.waiting_in_restaurant = True
            else:
                self.current_order.picked_up = True
                self.waiting_in_restaurant = False
                self.deliver_stage += 1
                self.last_restaurant = new_pos
                self.set_current_order()

        if (
            (len(self.order_list) > 0)
            and (self.path[self.deliver_stage][1] == new_pos)
            and (self.path[self.deliver_stage][0].split("_")[0] == "Customer")
        ):
            self.current_order.delivered = True
            self.current_order.delivered_at = self.model.steps
            self.on_location = True
            self.deliver_stage += 1

            # Check for delivery delay
            delay_time = self.current_order.delivered_at - self.current_order.deadline
            if delay_time > 0:
                # Find the customer and update their delay stats
                for agent in self.model.agents:
                    if (
                        isinstance(agent, Customer)
                        and agent.order_id == self.current_order.unique_id
                    ):
                        agent.register_delay(delay_time)
                        break
            else:
                for agent in self.model.agents:
                    if (
                        isinstance(agent, Customer)
                        and agent.order_id == self.current_order.unique_id
                    ):
                        agent.last_delay = 0
                        break

        # Передвигаем агента по карте на 1 клетку
        if not self.model.no_map:
            self.model.space.move_agent(self, new_pos)
        else:
            self.pos = new_pos

    def set_current_order(self):
        for agent in self.model.agents:
            if (isinstance(agent, Order)) and agent.unique_id == int(
                self.path[self.deliver_stage][0].split("_")[1]
            ):
                # Функция для удаления агента с карты
                self.current_order = agent
                break

    def find_closest_neighbor(self, agent_type):
        """
        Метод для поиска близжайшего заказа.
        Просто перебор
        """
        closest_agent = None
        min_distance = float("inf")

        for agent in self.model.agents:
            # Если заказ не поднят и у него нет назначенного курьера, то считаем до него дистанцию
            if (
                isinstance(agent, agent_type)
                and agent != self
                and not agent.picked_up
                and agent.courier_id == 0
            ):
                # Расстояние от курьера до ресторана
                distance = time_moving(
                    self.pos[0],
                    self.pos[1],
                    agent.restaurant_location[0],
                    agent.restaurant_location[1],
                )
                if distance < min_distance:
                    min_distance = distance
                    closest_agent = agent

        return closest_agent
