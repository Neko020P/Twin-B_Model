from mesa import Agent   
import random
import torch

class BaseAgent(Agent):
    def __init__(self, unique_id, model, zones,
                 preferred_temp=25.0,
                 comfort_tolerance=1.0,
                 initial_temp=None,
                 gender=None,
                 age=None,
                 route=None,
                 max_delta=3,
                 num_people_schedule=None,
                 activity_schedule=None,
                 agent_type=None,
                 **kwargs):

        super().__init__(unique_id, model)

        self.device = model.device
        self.agent_id = f"{agent_type}_{unique_id}" if agent_type else f"{self.__class__.__name__.lower()}_{unique_id}"
        self.agent_type = agent_type or self.__class__.__name__.replace("Agent", "").lower()
        self.current_room = random.choice(zones) if zones else "Unknown Zone"

        self.preferred_temp = preferred_temp
        self.comfort_tolerance = comfort_tolerance
        self.gender = gender
        self.age = age
        self.route = route if route is not None else []
        self.max_delta = max_delta

        if initial_temp is None:
            initial_temp = self.preferred_temp

        self.current_temp = torch.tensor(
            float(initial_temp),
            dtype=torch.float32,
            device=self.device
        )
        
        self.heat_gain_watts = kwargs.get("heat_gain_watts", 100.0)
        
        self.using_ac = False
        self.comfort_level = 0.0
        self.current_day = None
        self.current_hour = None

        self.in_room = True
        self.role = kwargs.get("role", None)

        self.num_people_schedule = num_people_schedule or []   
        self.activity_schedule = activity_schedule or []


    def step(self):
        temp = self.model.get_current_temp(self.current_room)
        if temp is None:
            temp = float('nan')
        
        if temp is not None and not (isinstance(temp, float) and temp != temp):
            self.current_temp = torch.tensor(float(temp), dtype=torch.float32, device=self.device)
        else:
            self.current_temp = torch.tensor(float('nan'), dtype=torch.float32, device=self.device)

        if temp is not None and not (isinstance(temp, float) and temp != temp):
            temp_val = float(self.current_temp) if torch.is_tensor(self.current_temp) else self.current_temp
            self.comfort_level = max(0.0, self.preferred_temp - abs(temp_val - self.preferred_temp))
        else:
            self.comfort_level = 0.0
        self.using_ac = abs(temp - self.preferred_temp) > self.comfort_tolerance

        self.current_day = getattr(self.model, "current_day", None)
        self.current_hour = getattr(self.model, "current_hour", None)

        self.model.agent_results.append({
            "day": self.current_day,
            "hour": self.current_hour,
            "step": self.model.current_step,
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "room": self.current_room,
            "current_temp": self.current_temp,
            "comfort_level": round(self.comfort_level, 2),
            "using_ac": self.using_ac,
            "preferred_temp": self.preferred_temp,
            "heat_gain_watts": self.heat_gain_watts
        })

class StudentUndergradAgent(BaseAgent): pass
class StudentGraduateAgent(BaseAgent): pass
class StaffAgent(BaseAgent): pass
class SupportStaffAgent(BaseAgent): pass
class VisitorAgent(BaseAgent): pass