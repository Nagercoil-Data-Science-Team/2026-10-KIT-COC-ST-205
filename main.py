

import os
import sys
import json
import argparse
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)


# ==============================================================================
# SECTION 1: SYSTEM & NETWORK CONFIGURATION (Step 2 Infrastructure)
# ==============================================================================

@dataclass
class EdgeServerConfig:
    server_id: str
    name: str
    cpu_cores: int
    cpu_freq_ghz: float
    queue_limit: int
    base_prop_delay_s: float
    uplink_bandwidth_mbps: float
    downlink_bandwidth_mbps: float


@dataclass
class CloudServerConfig:
    server_id: str = "Cloud_Server"
    name: str = "Central Cloud Datacenter"
    cpu_cores: int = 32
    cpu_freq_ghz: float = 3.8
    queue_limit: int = 1000
    base_prop_delay_s: float = 0.080  # 80ms WAN propagation latency
    uplink_bandwidth_mbps: float = 50.0
    downlink_bandwidth_mbps: float = 100.0


@dataclass
class NetworkConfig:
    protocol_overhead: float = 0.08  # 8% protocol overhead
    result_data_ratio: float = 0.05  # Output result size is 5% of input task size
    downlink_prop_delay_edge_s: float = 0.005  # 5ms downlink edge delay
    downlink_prop_delay_cloud_s: float = 0.060  # 60ms downlink cloud delay


@dataclass
class SystemConfig:
    num_iot_devices: int = 50
    edge_servers: List[EdgeServerConfig] = field(default_factory=lambda: [
        EdgeServerConfig("Edge_1", "Edge Server 1 (North)", 8, 3.0, 25, 0.008, 25.0, 50.0),
        EdgeServerConfig("Edge_2", "Edge Server 2 (Central)", 8, 3.2, 30, 0.010, 30.0, 60.0),
        EdgeServerConfig("Edge_3", "Edge Server 3 (South)", 8, 3.4, 25, 0.012, 25.0, 50.0),
    ])
    cloud_server: CloudServerConfig = field(default_factory=CloudServerConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


# ==============================================================================
# SECTION 2: DISCRETE-EVENT SIMULATION ENGINE CORE
# ==============================================================================

@dataclass
class TaskRecord:
    task_id: str
    device_id: str
    device_type: str
    workload_level: str
    arrival_time_s: float
    task_size_mb: float
    cpu_cycles_billion: float
    deadline_ms: float
    priority: str
    priority_level: int
    bandwidth_mbps: float
    local_cpu_ghz: float
    mobility: str
    destination: str = "Unassigned"
    dest_tier: str = "Unassigned"
    upload_start_s: float = 0.0
    upload_end_s: float = 0.0
    transmission_time_s: float = 0.0
    queue_enter_s: float = 0.0
    exec_start_s: float = 0.0
    queue_wait_time_s: float = 0.0
    exec_end_s: float = 0.0
    computation_time_s: float = 0.0
    return_start_s: float = 0.0
    return_end_s: float = 0.0
    return_time_s: float = 0.0
    completion_time_s: float = 0.0
    end_to_end_latency_s: float = 0.0
    end_to_end_latency_ms: float = 0.0
    slack_ms: float = 0.0
    deadline_met: bool = False
    deadline_status: str = "PENDING"
    queue_status: str = "SUCCESS"
    server_assigned: str = "None"
    core_id: int = -1


class ComputeNode:
    """Represents a computing node: Device, Edge Server, or Cloud Server."""
    def __init__(self, node_id: str, name: str, tier: str, num_cores: int, cpu_freq_ghz: float, queue_limit: int):
        self.node_id = node_id
        self.name = name
        self.tier = tier
        self.num_cores = num_cores
        self.cpu_freq_ghz = cpu_freq_ghz
        self.queue_limit = queue_limit
        self.core_available_at = [0.0] * num_cores
        self.waiting_queue: List[float] = []
        self.tasks_accepted = 0
        self.tasks_dropped = 0
        self.total_compute_time = 0.0
        self.queue_history: List[Tuple[float, int]] = [(0.0, 0)]
        self.utilization_history: List[Tuple[float, int]] = [(0.0, 0)]

    def record_queue_state(self, current_time: float):
        busy_cores = sum(1 for avail in self.core_available_at if avail > current_time)
        self.queue_history.append((current_time, len(self.waiting_queue)))
        self.utilization_history.append((current_time, busy_cores))

    def clean_waiting_queue(self, current_time: float) -> int:
        self.waiting_queue = [t for t in self.waiting_queue if t > current_time]
        return len(self.waiting_queue)


class MultiTierEdgeCloudSimulator:
    """Multi-Tier Edge-Cloud Simulator supporting Step 2 & Step 3 MDP environment."""
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or SystemConfig()
        self.reset()

    def reset(self):
        self.devices: Dict[str, ComputeNode] = {}
        self.edge_servers: Dict[str, ComputeNode] = {}
        self.nodes: Dict[str, ComputeNode] = {}
        
        for edge_cfg in self.config.edge_servers:
            node = ComputeNode(edge_cfg.server_id, edge_cfg.name, "Edge", edge_cfg.cpu_cores, edge_cfg.cpu_freq_ghz, edge_cfg.queue_limit)
            self.edge_servers[edge_cfg.server_id] = node
            self.nodes[edge_cfg.server_id] = node
            
        cloud_cfg = self.config.cloud_server
        self.cloud = ComputeNode(cloud_cfg.server_id, cloud_cfg.name, "Cloud", cloud_cfg.cpu_cores, cloud_cfg.cpu_freq_ghz, cloud_cfg.queue_limit)
        self.nodes[cloud_cfg.server_id] = self.cloud
        
        self.active_transmissions: List[Tuple[float, float, str]] = []
        self.task_records: List[TaskRecord] = []

    def get_or_create_device(self, device_id: str, device_type: str, local_cpu_ghz: float) -> ComputeNode:
        if device_id not in self.devices:
            dev = ComputeNode(device_id, f"IoT_{device_id}", "Device", 1, local_cpu_ghz, 10)
            self.devices[device_id] = dev
            self.nodes[device_id] = dev
        return self.devices[device_id]

    def calculate_transmission_delay(self, task_size_mb: float, bandwidth_mbps: float, base_prop_delay_s: float, arrival_time_s: float, link_id: str) -> float:
        ideal_tx = (task_size_mb * 8.0) / max(bandwidth_mbps, 0.1)
        protocol_mult = 1.0 + self.config.network.protocol_overhead
        concurrent = sum(1 for (s, e, lk) in self.active_transmissions if lk == link_id and s <= arrival_time_s <= e)
        congestion_penalty = 1.0 + 0.05 * min(concurrent, 10)
        actual_tx = (ideal_tx * protocol_mult * congestion_penalty) + base_prop_delay_s
        self.active_transmissions.append((arrival_time_s, arrival_time_s + actual_tx, link_id))
        return actual_tx

    def calculate_return_delay(self, task_size_mb: float, downlink_bw_mbps: float, downlink_prop_delay_s: float) -> float:
        result_size_mb = task_size_mb * self.config.network.result_data_ratio
        ideal_rx = (result_size_mb * 8.0) / max(downlink_bw_mbps, 1.0)
        return ideal_rx + downlink_prop_delay_s

    def schedule_execution_on_node(self, node: ComputeNode, arrival_at_server_s: float, cpu_cycles_billion: float) -> Tuple[bool, float, float, float, int]:
        node.waiting_queue = [t for t in node.waiting_queue if t > arrival_at_server_s]
        node.record_queue_state(arrival_at_server_s)
        
        if len(node.waiting_queue) >= node.queue_limit:
            node.tasks_dropped += 1
            node.record_queue_state(arrival_at_server_s)
            return False, 0.0, 0.0, 0.0, -1
            
        node.tasks_accepted += 1
        computation_s = cpu_cycles_billion / node.cpu_freq_ghz
        earliest_core_idx = int(np.argmin(node.core_available_at))
        core_free_at = node.core_available_at[earliest_core_idx]
        
        exec_start_s = max(arrival_at_server_s, core_free_at)
        queue_wait_s = exec_start_s - arrival_at_server_s
        exec_end_s = exec_start_s + computation_s
        
        if exec_start_s > arrival_at_server_s:
            node.waiting_queue.append(exec_start_s)
            
        node.core_available_at[earliest_core_idx] = exec_end_s
        node.total_compute_time += computation_s
        node.record_queue_state(exec_start_s)
        return True, queue_wait_s, exec_start_s, computation_s, earliest_core_idx

    def simulate_task(self, task_data: Dict[str, Any], offload_decision: str) -> TaskRecord:
        arrival_s = float(task_data["Arrival_Time_s"])
        task_size_mb = float(task_data["Task_Size_MB"])
        cpu_billion = float(task_data["CPU_Cycles_Billion"])
        deadline_ms = float(task_data["Deadline_ms"])
        bandwidth_mbps = float(task_data["Bandwidth_Mbps"])
        local_cpu_ghz = float(task_data["Local_CPU_GHz"])
        device_id = str(task_data["Device_ID"])
        device_type = str(task_data["Device_Type"])

        record = TaskRecord(
            task_id=str(task_data["Task_ID"]),
            device_id=device_id,
            device_type=device_type,
            workload_level=str(task_data.get("Workload_Level", "Medium")),
            arrival_time_s=arrival_s,
            task_size_mb=task_size_mb,
            cpu_cycles_billion=cpu_billion,
            deadline_ms=deadline_ms,
            priority=str(task_data.get("Priority", "Normal")),
            priority_level=int(task_data.get("Priority_Level", 2)),
            bandwidth_mbps=bandwidth_mbps,
            local_cpu_ghz=local_cpu_ghz,
            mobility=str(task_data.get("Mobility", "Fixed")),
            destination=offload_decision,
        )

        if offload_decision == "Local":
            record.dest_tier = "Device"
            record.server_assigned = f"Local_{device_id}"
            device_node = self.get_or_create_device(device_id, device_type, local_cpu_ghz)
            accepted, q_wait, ex_start, comp_s, core_id = self.schedule_execution_on_node(device_node, arrival_s, cpu_billion)
            if not accepted:
                record.queue_status = "REJECTED_QUEUE_OVERFLOW"
                record.deadline_met = False
                record.deadline_status = "VIOLATED"
                record.end_to_end_latency_s = (deadline_ms / 1000.0) * 2.0
                record.end_to_end_latency_ms = record.end_to_end_latency_s * 1000.0
                record.completion_time_s = arrival_s + record.end_to_end_latency_s
                record.slack_ms = -deadline_ms
                self.task_records.append(record)
                return record

            record.queue_wait_time_s = q_wait
            record.computation_time_s = comp_s
            record.completion_time_s = ex_start + comp_s
            record.end_to_end_latency_s = record.completion_time_s - arrival_s
            record.end_to_end_latency_ms = record.end_to_end_latency_s * 1000.0
            record.slack_ms = record.deadline_ms - record.end_to_end_latency_ms
            record.deadline_met = (record.end_to_end_latency_ms <= record.deadline_ms)
            record.deadline_status = "MET" if record.deadline_met else "VIOLATED"
            self.task_records.append(record)
            return record

        elif offload_decision.startswith("Edge_"):
            record.dest_tier = "Edge"
            edge_cfg = next((c for c in self.config.edge_servers if c.server_id == offload_decision), self.config.edge_servers[0])
            edge_node = self.edge_servers[edge_cfg.server_id]
            record.server_assigned = edge_cfg.server_id
            
            tx_time = self.calculate_transmission_delay(task_size_mb, bandwidth_mbps, edge_cfg.base_prop_delay_s, arrival_s, f"{device_id}_{edge_cfg.server_id}")
            record.transmission_time_s = tx_time
            arrival_at_server = arrival_s + tx_time
            
            accepted, q_wait, ex_start, comp_s, core_id = self.schedule_execution_on_node(edge_node, arrival_at_server, cpu_billion)
            if not accepted:
                record.queue_status = "REJECTED_QUEUE_OVERFLOW"
                record.deadline_met = False
                record.deadline_status = "VIOLATED"
                record.end_to_end_latency_s = (deadline_ms / 1000.0) * 2.0
                record.end_to_end_latency_ms = record.end_to_end_latency_s * 1000.0
                record.completion_time_s = arrival_s + record.end_to_end_latency_s
                record.slack_ms = -deadline_ms
                self.task_records.append(record)
                return record

            rx_time = self.calculate_return_delay(task_size_mb, edge_cfg.downlink_bandwidth_mbps, self.config.network.downlink_prop_delay_edge_s)
            record.queue_wait_time_s = q_wait
            record.computation_time_s = comp_s
            record.return_time_s = rx_time
            record.completion_time_s = ex_start + comp_s + rx_time
            record.end_to_end_latency_s = record.completion_time_s - arrival_s
            record.end_to_end_latency_ms = record.end_to_end_latency_s * 1000.0
            record.slack_ms = record.deadline_ms - record.end_to_end_latency_ms
            record.deadline_met = (record.end_to_end_latency_ms <= record.deadline_ms)
            record.deadline_status = "MET" if record.deadline_met else "VIOLATED"
            self.task_records.append(record)
            return record

        elif offload_decision == "Cloud_Server" or offload_decision == "Cloud":
            record.dest_tier = "Cloud"
            cloud_cfg = self.config.cloud_server
            record.server_assigned = cloud_cfg.server_id
            
            tx_time = self.calculate_transmission_delay(task_size_mb, cloud_cfg.uplink_bandwidth_mbps, cloud_cfg.base_prop_delay_s, arrival_s, "Cloud_WAN")
            record.transmission_time_s = tx_time
            arrival_at_server = arrival_s + tx_time
            
            accepted, q_wait, ex_start, comp_s, core_id = self.schedule_execution_on_node(self.cloud, arrival_at_server, cpu_billion)
            if not accepted:
                record.queue_status = "REJECTED_QUEUE_OVERFLOW"
                record.deadline_met = False
                record.deadline_status = "VIOLATED"
                record.end_to_end_latency_s = (deadline_ms / 1000.0) * 2.0
                record.end_to_end_latency_ms = record.end_to_end_latency_s * 1000.0
                record.completion_time_s = arrival_s + record.end_to_end_latency_s
                record.slack_ms = -deadline_ms
                self.task_records.append(record)
                return record

            rx_time = self.calculate_return_delay(task_size_mb, cloud_cfg.downlink_bandwidth_mbps, self.config.network.downlink_prop_delay_cloud_s)
            record.queue_wait_time_s = q_wait
            record.computation_time_s = comp_s
            record.return_time_s = rx_time
            record.completion_time_s = ex_start + comp_s + rx_time
            record.end_to_end_latency_s = record.completion_time_s - arrival_s
            record.end_to_end_latency_ms = record.end_to_end_latency_s * 1000.0
            record.slack_ms = record.deadline_ms - record.end_to_end_latency_ms
            record.deadline_met = (record.end_to_end_latency_ms <= record.deadline_ms)
            record.deadline_status = "MET" if record.deadline_met else "VIOLATED"
            self.task_records.append(record)
            return record
        else:
            raise ValueError(f"Invalid offloading destination: {offload_decision}")


# ==============================================================================
# SECTION 3: STEP 3 - STATE–ACTION–REWARD FORMULATION (MDP DESIGN)
# ==============================================================================

class ActionSpaceManager:
    """
    Step 3.2 Action Space & Infeasible Action Handling:
    Discrete Actions:
      Action 0: Local execution on IoT Device.
      Action 1: Edge Server 1 (North).
      Action 2: Edge Server 2 (Central).
      Action 3: Edge Server 3 (South).
      Action 4: Remote Cloud Server.
    """
    ACTION_NAMES = ["Local", "Edge_1", "Edge_2", "Edge_3", "Cloud_Server"]
    NUM_ACTIONS = 5

    @classmethod
    def get_action_mask(cls, sim: MultiTierEdgeCloudSimulator, device_id: str, arrival_time_s: float) -> np.ndarray:
        """
        Step 3.4 Infeasible Action Masking:
        Returns binary mask [m_0, m_1, m_2, m_3, m_4] where:
          1 = Action is feasible (destination queue has space).
          0 = Action is infeasible (server is overloaded/full).
        """
        mask = np.ones(cls.NUM_ACTIONS, dtype=np.float32)

        # 0. Local Device Queue Check
        dev = sim.devices.get(device_id)
        if dev is not None:
            dev_q = dev.clean_waiting_queue(arrival_time_s)
            if dev_q >= dev.queue_limit:
                mask[0] = 0.0

        # 1-3. Edge Server Queues Check
        for idx, edge_id in enumerate(["Edge_1", "Edge_2", "Edge_3"], start=1):
            edge_node = sim.edge_servers.get(edge_id)
            if edge_node is not None:
                q_len = edge_node.clean_waiting_queue(arrival_time_s)
                if q_len >= edge_node.queue_limit:
                    mask[idx] = 0.0

        # 4. Cloud Server Queue Check
        if sim.cloud is not None:
            c_q = sim.cloud.clean_waiting_queue(arrival_time_s)
            if c_q >= sim.cloud.queue_limit:
                mask[4] = 0.0

        # Safety: at least one fallback action must be unmasked
        if np.sum(mask) == 0:
            mask[4] = 1.0  # Cloud acts as ultimate fallback
        return mask

    @classmethod
    def resolve_action(cls, action_idx: int, action_mask: np.ndarray) -> Tuple[str, bool, str]:
        """
        Validates action feasibility. If infeasible, triggers Step 3.4 fallback redirect.
        Returns: (final_destination, was_infeasible, fallback_reason)
        """
        dest = cls.ACTION_NAMES[action_idx]
        if action_mask[action_idx] == 1.0:
            return dest, False, "Feasible execution"
        
        # Infeasible action handling: fallback to least congested feasible tier
        feasible_indices = np.where(action_mask == 1.0)[0]
        # Prefer edge servers if available, else Cloud
        fallback_idx = feasible_indices[-1] if len(feasible_indices) > 0 else 4
        fallback_dest = cls.ACTION_NAMES[fallback_idx]
        return fallback_dest, True, f"Infeasible destination {dest} redirected to {fallback_dest}"


class StateObservationManager:
    """
    Step 3.1 State / Observation Space Formulation:
    Extracts normalized local observation for actor and global state for critic.
    Features:
      1. Task CPU Demand (Cycles in Billions normalized / 2.0)
      2. Task Data Size (MB normalized / 5.0)
      3. Remaining Deadline (seconds normalized / 1.0)
      4. Local CPU Utilization (busy fraction in [0, 1])
      5. Edge 1 Queue Length (normalized / queue_limit)
      6. Edge 2 Queue Length (normalized / queue_limit)
      7. Edge 3 Queue Length (normalized / queue_limit)
      8. Cloud Queue Length (normalized / 100.0)
      9. Network Bandwidth (Mbps normalized / 100.0)
      10. Estimated Comm Delay to Edge (seconds)
      11. Estimated Comm Delay to Cloud (seconds)
      12. Task Priority Level (normalized / 4.0)
    """
    FEATURE_NAMES = [
        "Task_CPU_Demand_Norm",
        "Task_Data_Size_Norm",
        "Remaining_Deadline_Norm",
        "Local_CPU_Util",
        "Edge_1_Queue_Norm",
        "Edge_2_Queue_Norm",
        "Edge_3_Queue_Norm",
        "Cloud_Queue_Norm",
        "Network_Bandwidth_Norm",
        "Est_Comm_Delay_Edge_s",
        "Est_Comm_Delay_Cloud_s",
        "Task_Priority_Norm",
    ]
    OBS_DIM = len(FEATURE_NAMES)

    @classmethod
    def extract_observation(cls, sim: MultiTierEdgeCloudSimulator, task_dict: Dict[str, Any]) -> np.ndarray:
        arr_time = float(task_dict["Arrival_Time_s"])
        task_size_mb = float(task_dict["Task_Size_MB"])
        cpu_billion = float(task_dict["CPU_Cycles_Billion"])
        deadline_s = float(task_dict["Deadline_ms"]) / 1000.0
        bw_mbps = float(task_dict["Bandwidth_Mbps"])
        dev_id = str(task_dict["Device_ID"])
        prio_lvl = float(task_dict.get("Priority_Level", 2))

        # Local device utilization
        dev = sim.devices.get(dev_id)
        local_busy = 0.0
        if dev is not None:
            dev.clean_waiting_queue(arr_time)
            local_busy = 1.0 if dev.core_available_at[0] > arr_time else 0.0

        # Edge queues
        q_e1 = sim.edge_servers["Edge_1"].clean_waiting_queue(arr_time) / sim.edge_servers["Edge_1"].queue_limit
        q_e2 = sim.edge_servers["Edge_2"].clean_waiting_queue(arr_time) / sim.edge_servers["Edge_2"].queue_limit
        q_e3 = sim.edge_servers["Edge_3"].clean_waiting_queue(arr_time) / sim.edge_servers["Edge_3"].queue_limit
        q_cloud = sim.cloud.clean_waiting_queue(arr_time) / 100.0

        # Estimated transmission delays
        est_edge_tx = (task_size_mb * 8.0 / max(bw_mbps, 0.1)) * 1.08 + 0.010
        est_cloud_tx = (task_size_mb * 8.0 / 50.0) * 1.08 + 0.080

        obs = np.array([
            cpu_billion / 2.0,            # Feature 1
            task_size_mb / 5.0,           # Feature 2
            min(deadline_s, 2.0) / 2.0,   # Feature 3
            local_busy,                   # Feature 4
            min(q_e1, 1.0),               # Feature 5
            min(q_e2, 1.0),               # Feature 6
            min(q_e3, 1.0),               # Feature 7
            min(q_cloud, 1.0),            # Feature 8
            min(bw_mbps, 100.0) / 100.0,  # Feature 9
            min(est_edge_tx, 5.0) / 5.0,  # Feature 10
            min(est_cloud_tx, 5.0) / 5.0, # Feature 11
            prio_lvl / 4.0,               # Feature 12
        ], dtype=np.float32)

        return obs

    @classmethod
    def extract_global_state(cls, sim: MultiTierEdgeCloudSimulator, task_dict: Dict[str, Any]) -> np.ndarray:
        """Global critic state for Centralized Training with Decentralized Execution (CTDE)."""
        local_obs = cls.extract_observation(sim, task_dict)
        arr_time = float(task_dict["Arrival_Time_s"])
        
        # Global system metrics
        edge_active_cores = [
            sum(1 for avail in srv.core_available_at if avail > arr_time)
            for srv in sim.edge_servers.values()
        ]
        cloud_active_cores = sum(1 for avail in sim.cloud.core_available_at if avail > arr_time)
        active_transmissions = len([1 for (s, e, _) in sim.active_transmissions if s <= arr_time <= e])

        extra_global = np.array([
            edge_active_cores[0] / 8.0,
            edge_active_cores[1] / 8.0,
            edge_active_cores[2] / 8.0,
            cloud_active_cores / 32.0,
            min(active_transmissions, 20) / 20.0,
        ], dtype=np.float32)

        return np.concatenate([local_obs, extra_global])


class RewardFunctionManager:
    """
    Step 3.3 Reward Function:
      r_t = - [ alpha * (D_t / D_ref) + beta * M_t + lambda * C_t ] + R_bonus - Infeasible_Penalty

    Where:
      - D_t: Measured task end-to-end latency.
      - D_ref: Normalization reference for latency (Task Deadline).
      - M_t: Deadline violation penalty (1 if violated/dropped, 0 if met).
      - C_t: Congestion penalty (queue occupancy of target node).
      - R_bonus: Positive bonus if deadline is met.
      - Infeasible_Penalty: Penalty if agent chose an overloaded/infeasible destination.
    """
    def __init__(
        self,
        alpha: float = 1.0,   # Latency weight
        beta: float = 2.0,    # Deadline violation weight
        lambda_c: float = 0.5,# Congestion weight
        bonus: float = 1.0,   # Positive reward for on-time completion
        infeasible_penalty: float = 2.5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.lambda_c = lambda_c
        self.bonus = bonus
        self.infeasible_penalty = infeasible_penalty

    def compute_reward(
        self,
        record: TaskRecord,
        target_node: ComputeNode,
        was_infeasible: bool,
    ) -> Tuple[float, Dict[str, float]]:
        # 1. Normalized Latency Term
        d_t = record.end_to_end_latency_ms
        d_ref = max(record.deadline_ms, 1.0)
        norm_latency = d_t / d_ref
        latency_penalty = self.alpha * norm_latency

        # 2. Deadline Violation Term
        m_t = 1.0 if (not record.deadline_met or record.queue_status != "SUCCESS") else 0.0
        deadline_penalty = self.beta * m_t

        # 3. Congestion Term
        q_len = len(target_node.waiting_queue)
        q_limit = max(target_node.queue_limit, 1)
        c_t = min(q_len / q_limit, 1.0)
        congestion_penalty = self.lambda_c * c_t

        # 4. Success Bonus
        bonus_term = self.bonus if (record.deadline_met and record.queue_status == "SUCCESS") else 0.0

        # 5. Infeasible action penalty
        infeasible_term = self.infeasible_penalty if was_infeasible else 0.0

        # Total reward formula
        reward = -(latency_penalty + deadline_penalty + congestion_penalty) + bonus_term - infeasible_term

        breakdown = {
            "Latency_Penalty": float(latency_penalty),
            "Deadline_Penalty": float(deadline_penalty),
            "Congestion_Penalty": float(congestion_penalty),
            "Bonus_Reward": float(bonus_term),
            "Infeasible_Penalty": float(infeasible_term),
            "Total_Reward": float(reward),
            "Normalized_Latency_Dt_Dref": float(norm_latency),
            "Deadline_Violation_Mt": float(m_t),
            "Congestion_Ct": float(c_t),
        }
        return float(reward), breakdown


# ==============================================================================
# SECTION 4: GYM-COMPATIBLE MULTI-TIER RL ENVIRONMENT
# ==============================================================================

class MultiTierEdgeCloudEnv:
    """
    OpenAI Gym-style Environment for Step 3 MDP Execution.
    Provides standard reset() and step(action) methods.
    """
    def __init__(self, workload_df: pd.DataFrame, config: Optional[SystemConfig] = None):
        self.workload_df = workload_df.sort_values("Arrival_Time_s").reset_index(drop=True)
        self.total_tasks = len(self.workload_df)
        self.config = config or SystemConfig()
        self.sim = MultiTierEdgeCloudSimulator(self.config)
        self.reward_mgr = RewardFunctionManager()
        self.current_step = 0

    def reset(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        self.sim.reset()
        self.current_step = 0
        task_data = self.workload_df.iloc[0].to_dict()
        obs = StateObservationManager.extract_observation(self.sim, task_data)
        mask = ActionSpaceManager.get_action_mask(self.sim, str(task_data["Device_ID"]), float(task_data["Arrival_Time_s"]))
        info = {"task_id": task_data["Task_ID"], "global_state": StateObservationManager.extract_global_state(self.sim, task_data)}
        return obs, mask, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        task_data = self.workload_df.iloc[self.current_step].to_dict()
        arr_time = float(task_data["Arrival_Time_s"])
        dev_id = str(task_data["Device_ID"])

        # Step 3.4 Infeasible Action Handling
        mask = ActionSpaceManager.get_action_mask(self.sim, dev_id, arr_time)
        dest, was_infeasible, reason = ActionSpaceManager.resolve_action(action_idx, mask)

        # Execute simulation step
        record = self.sim.simulate_task(task_data, offload_decision=dest)
        target_node = self.sim.nodes.get(dest, self.sim.cloud)

        # Step 3.3 Compute Reward
        reward, breakdown = self.reward_mgr.compute_reward(record, target_node, was_infeasible)

        self.current_step += 1
        done = (self.current_step >= self.total_tasks)
        truncated = False

        if not done:
            next_task = self.workload_df.iloc[self.current_step].to_dict()
            next_obs = StateObservationManager.extract_observation(self.sim, next_task)
            next_mask = ActionSpaceManager.get_action_mask(self.sim, str(next_task["Device_ID"]), float(next_task["Arrival_Time_s"]))
            next_global = StateObservationManager.extract_global_state(self.sim, next_task)
        else:
            next_obs = np.zeros(StateObservationManager.OBS_DIM, dtype=np.float32)
            next_mask = np.ones(ActionSpaceManager.NUM_ACTIONS, dtype=np.float32)
            next_global = np.zeros(StateObservationManager.OBS_DIM + 5, dtype=np.float32)

        info = {
            "record": record,
            "reward_breakdown": breakdown,
            "was_infeasible": was_infeasible,
            "resolution_reason": reason,
            "action_mask": mask,
            "chosen_action": action_idx,
            "destination": dest,
            "global_state": next_global,
        }
        return next_obs, reward, done, truncated, info


# ==============================================================================
# SECTION 5: MDP EXECUTION & TRANSITION DATASET GENERATION
# ==============================================================================

def run_mdp_simulation(
    workload_df: pd.DataFrame,
    policy_type: str = "Queue_Aware_Adaptive",
) -> Tuple[pd.DataFrame, pd.DataFrame, MultiTierEdgeCloudSimulator]:
    """
    Executes the Step 3 MDP environment with the specified policy.
    Policies:
      - 'Queue_Aware_Adaptive': Adaptive policy picking feasible action minimizing estimated delay.
      - 'Random_Policy': Random agent exploring action space with masking.
      - 'Local_Only': Baseline agent selecting Action 0.
      - 'Cloud_Only': Baseline agent selecting Action 4.
      - 'Edge_Greedy': Baseline agent selecting Action 1-3.
    """
    env = MultiTierEdgeCloudEnv(workload_df)
    obs, mask, info = env.reset()
    
    transition_rows = []
    task_records = []
    done = False
    step_idx = 0

    while not done:
        task_data = workload_df.iloc[step_idx].to_dict()
        arr_time = float(task_data["Arrival_Time_s"])

        # Policy decision
        if policy_type == "Local_Only":
            action = 0
        elif policy_type == "Cloud_Only":
            action = 4
        elif policy_type == "Edge_Greedy":
            # Best edge server
            edge_indices = [1, 2, 3]
            feasible_edges = [i for i in edge_indices if mask[i] == 1.0]
            action = feasible_edges[step_idx % len(feasible_edges)] if feasible_edges else 4
        elif policy_type == "Random_Policy":
            feasible_actions = np.where(mask == 1.0)[0]
            action = int(np.random.choice(feasible_actions))
        else:  # Queue_Aware_Adaptive (Heuristic towards Step 4 QTM-MAPPO)
            best_action = 0
            min_score = float("inf")
            for a_idx in range(ActionSpaceManager.NUM_ACTIONS):
                dest_name = ActionSpaceManager.ACTION_NAMES[a_idx]
                cpu_b = float(task_data["CPU_Cycles_Billion"])
                if dest_name == "Local":
                    est = cpu_b / float(task_data["Local_CPU_GHz"])
                elif dest_name.startswith("Edge_"):
                    tx = (float(task_data["Task_Size_MB"]) * 8.0 / max(float(task_data["Bandwidth_Mbps"]), 0.1)) * 1.08 + 0.010
                    comp = cpu_b / env.sim.edge_servers[dest_name].cpu_freq_ghz
                    q_pen = 5.0 if mask[a_idx] == 0.0 else 0.0
                    est = tx + comp + q_pen
                else:  # Cloud
                    tx = (float(task_data["Task_Size_MB"]) * 8.0 / 50.0) * 1.08 + 0.080
                    comp = cpu_b / 3.8
                    est = tx + comp
                if est < min_score:
                    min_score = est
                    best_action = a_idx
            action = best_action

        # Step environment
        next_obs, reward, done, _, step_info = env.step(action)
        rec: TaskRecord = step_info["record"]
        bd = step_info["reward_breakdown"]

        transition_rows.append({
            "Step": step_idx,
            "Task_ID": rec.task_id,
            "Device_ID": rec.device_id,
            "Arrival_Time_s": rec.arrival_time_s,
            "Chosen_Action": action,
            "Destination": rec.destination,
            "Dest_Tier": rec.dest_tier,
            "Was_Infeasible": step_info["was_infeasible"],
            "Reward": round(reward, 4),
            "Latency_Penalty": round(bd["Latency_Penalty"], 4),
            "Deadline_Penalty": round(bd["Deadline_Penalty"], 4),
            "Congestion_Penalty": round(bd["Congestion_Penalty"], 4),
            "Bonus_Reward": round(bd["Bonus_Reward"], 4),
            "Infeasible_Penalty": round(bd["Infeasible_Penalty"], 4),
            "End_to_End_Latency_ms": round(rec.end_to_end_latency_ms, 2),
            "Deadline_ms": round(rec.deadline_ms, 2),
            "Slack_ms": round(rec.slack_ms, 2),
            "Deadline_Status": rec.deadline_status,
            "Queue_Wait_ms": round(rec.queue_wait_time_s * 1000, 2),
            "Transmission_ms": round(rec.transmission_time_s * 1000, 2),
            "Computation_ms": round(rec.computation_time_s * 1000, 2),
            "Return_ms": round(rec.return_time_s * 1000, 2),
            # Key State Features
            "State_Task_CPU_Norm": round(float(obs[0]), 4),
            "State_Task_Size_Norm": round(float(obs[1]), 4),
            "State_Deadline_Norm": round(float(obs[2]), 4),
            "State_Local_Busy": round(float(obs[3]), 4),
            "State_Edge1_Queue": round(float(obs[4]), 4),
            "State_Edge2_Queue": round(float(obs[5]), 4),
            "State_Edge3_Queue": round(float(obs[6]), 4),
            "State_Cloud_Queue": round(float(obs[7]), 4),
            "State_Bandwidth_Norm": round(float(obs[8]), 4),
            "Mask_Local": int(mask[0]),
            "Mask_Edge1": int(mask[1]),
            "Mask_Edge2": int(mask[2]),
            "Mask_Edge3": int(mask[3]),
            "Mask_Cloud": int(mask[4]),
        })
        task_records.append(asdict(rec))
        obs = next_obs
        step_idx += 1

    df_transitions = pd.DataFrame(transition_rows)
    df_records = pd.DataFrame(task_records)
    return df_transitions, df_records, env.sim


# ==============================================================================
# SECTION 6: VISUALIZATION MODULE (5 SEPARATE WINDOWS)
# ==============================================================================

def set_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 150,
    })


def plot_window_1_state_distribution(df: pd.DataFrame, save_path: str):
    """Window 1: RL State / Observation Feature Distribution."""
    fig = plt.figure(num="Window 1: RL State Feature Space Distribution", figsize=(12, 6))
    
    state_cols = [
        ("State_Task_CPU_Norm", "Task CPU Demand (Norm)", "#1d3557"),
        ("State_Task_Size_Norm", "Task Data Size (Norm)", "#457b9d"),
        ("State_Deadline_Norm", "Remaining Deadline (Norm)", "#2a9d8f"),
        ("State_Bandwidth_Norm", "Network Bandwidth (Norm)", "#e76f51"),
    ]
    
    for idx, (col, title, color) in enumerate(state_cols, start=1):
        ax = fig.add_subplot(2, 2, idx)
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=25, alpha=0.6)
        ax.set_title(title, fontsize=11, pad=6)
        ax.set_xlabel("Normalized Value [0, 1]", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figure 1 (Step 3.1): State / Observation Feature Distributions", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def plot_window_2_action_dynamics(df: pd.DataFrame, save_path: str):
    """Window 2: Action Space Selection & Feasibility Masking Dynamics."""
    fig = plt.figure(num="Window 2: Action Selection & Masking Dynamics", figsize=(12, 5.5))
    
    # Subplot 1: Action Distribution
    ax1 = fig.add_subplot(1, 2, 1)
    action_counts = df["Destination"].value_counts()
    colors = ["#2b5c8f", "#2a9d8f", "#457b9d", "#f4a261", "#e76f51"]
    bars = ax1.bar(action_counts.index, action_counts.values, color=colors[:len(action_counts)], width=0.5)
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f"{int(bar.get_height())}", ha="center", va="bottom", fontweight="bold")
    ax1.set_title("Step 3.2: Offloading Action Frequencies", pad=10)
    ax1.set_xlabel("Offloading Destination Action", labelpad=8)
    ax1.set_ylabel("Execution Count", labelpad=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Action Feasibility Dynamics
    ax2 = fig.add_subplot(1, 2, 2)
    mask_means = {
        "Local": df["Mask_Local"].mean() * 100,
        "Edge 1": df["Mask_Edge1"].mean() * 100,
        "Edge 2": df["Mask_Edge2"].mean() * 100,
        "Edge 3": df["Mask_Edge3"].mean() * 100,
        "Cloud": df["Mask_Cloud"].mean() * 100,
    }
    bars2 = ax2.bar(mask_means.keys(), mask_means.values(), color="#2a9d8f", width=0.5)
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.1f}%", ha="center", va="bottom", fontweight="bold")
    ax2.set_title("Step 3.4: Action Feasibility Rate (Action Mask == 1)", pad=10)
    ax2.set_ylabel("Feasibility Rate (%)", labelpad=8)
    ax2.set_ylim(0, 115)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figure 2: Action Space Selection & Feasibility Masking", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def plot_window_3_reward_breakdown(df: pd.DataFrame, save_path: str):
    """Window 3: Step 3.3 Reward Trajectory and Penalty Component Breakdown."""
    fig = plt.figure(num="Window 3: Reward Formulation & Penalty Breakdown", figsize=(12, 6))
    
    # Subplot 1: Cumulative & Rolling Reward
    ax1 = fig.add_subplot(2, 1, 1)
    df["Rolling_Reward"] = df["Reward"].rolling(window=30, min_periods=1).mean()
    ax1.plot(df["Step"], df["Reward"], color="#b0c4de", alpha=0.4, label="Instantaneous Step Reward r_t")
    ax1.plot(df["Step"], df["Rolling_Reward"], color="#1d3557", linewidth=2.0, label="30-Step Rolling Mean Reward")
    ax1.set_title("Figure 3(a): Reward Trajectory over Simulation Tasks", pad=8)
    ax1.set_ylabel("Reward Value", labelpad=8)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Penalty Components Breakdown
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df["Step"], df["Latency_Penalty"], label="Latency Penalty (alpha * D_t/D_ref)", color="#457b9d", linewidth=1.5)
    ax2.plot(df["Step"], df["Deadline_Penalty"], label="Deadline Penalty (beta * M_t)", color="#e63946", linewidth=1.5)
    ax2.plot(df["Step"], df["Congestion_Penalty"], label="Congestion Penalty (lambda * C_t)", color="#f4a261", linewidth=1.5)
    ax2.set_title("Figure 3(b): Step 3.3 Normalized Reward Penalty Components", pad=8)
    ax2.set_xlabel("Simulation Task Step (t)", labelpad=8)
    ax2.set_ylabel("Penalty Magnitude", labelpad=8)
    ax2.legend(loc="upper right", ncol=3, fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def plot_window_4_latency_and_slack(df: pd.DataFrame, save_path: str):
    """Window 4: End-to-End Latency vs Deadline Slack & Cumulative Reward."""
    fig = plt.figure(num="Window 4: Latency vs Deadline Slack Analysis", figsize=(12, 5.5))
    
    # Subplot 1: Latency by Destination Tier Boxplot
    ax1 = fig.add_subplot(1, 2, 1)
    palette = {"Device": "#2b5c8f", "Edge": "#2a9d8f", "Cloud": "#e76f51"}
    sns.boxplot(
        data=df,
        x="Dest_Tier",
        y="End_to_End_Latency_ms",
        hue="Dest_Tier",
        legend=False,
        palette=palette,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "yellow", "markeredgecolor": "black", "markersize": 7},
        ax=ax1,
        width=0.45,
    )
    mean_deadline = df["Deadline_ms"].mean()
    ax1.axhline(mean_deadline, color="red", linestyle="--", linewidth=1.8, label=f"Mean Deadline ({mean_deadline:.1f} ms)")
    ax1.set_title("End-to-End Latency by Destination Tier", pad=10)
    ax1.set_xlabel("Destination Tier", labelpad=8)
    ax1.set_ylabel("Latency (ms)", labelpad=8)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Cumulative Reward vs Task Index
    ax2 = fig.add_subplot(1, 2, 2)
    cum_reward = df["Reward"].cumsum()
    ax2.plot(df["Step"], cum_reward, color="#2a9d8f", linewidth=2.2, label="Cumulative MDP Reward")
    ax2.set_title("Cumulative Episodic Reward Progression", pad=10)
    ax2.set_xlabel("Simulation Task Step (t)", labelpad=8)
    ax2.set_ylabel("Cumulative Reward Sum", labelpad=8)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figure 4: Latency Performance & Cumulative Reward Progression", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def plot_window_5_policy_benchmark(policy_results: Dict[str, pd.DataFrame], save_path: str):
    """Window 5: Offloading Policy Comparison (Reward, Latency, Deadline Miss Ratio)."""
    fig = plt.figure(num="Window 5: Policy Benchmark Comparison", figsize=(12, 6))
    
    policies = list(policy_results.keys())
    mean_rewards = [res["Reward"].mean() for res in policy_results.values()]
    mean_latencies = [res["End_to_End_Latency_ms"].mean() for res in policy_results.values()]
    miss_rates = [(res["Deadline_Status"] == "VIOLATED").mean() * 100 for res in policy_results.values()]

    # Subplot 1: Mean Reward
    ax1 = fig.add_subplot(1, 2, 1)
    colors = ["#2a9d8f" if r > -1.5 else "#f4a261" if r > -2.5 else "#e63946" for r in mean_rewards]
    bars1 = ax1.bar(policies, mean_rewards, color=colors, width=0.45)
    for bar, val in zip(bars1, mean_rewards):
        ax1.text(bar.get_x() + bar.get_width()/2, val - 0.15 if val < 0 else val + 0.05, f"{val:.2f}", ha="center", va="top" if val < 0 else "bottom", fontweight="bold", fontsize=9)
    ax1.set_xticks(range(len(policies)))
    ax1.set_xticklabels(policies, rotation=25, ha="right", fontsize=9)
    ax1.set_title("Mean Step Reward (Higher is Better)", pad=10)
    ax1.set_ylabel("Average Reward per Task", labelpad=8)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Latency & Deadline-Miss Ratio
    ax2 = fig.add_subplot(1, 2, 2)
    x = np.arange(len(policies))
    width = 0.35
    ax2.bar(x - width/2, mean_latencies, width, label="Mean Latency (ms)", color="#457b9d")
    ax3 = ax2.twinx()
    ax3.plot(x, miss_rates, color="#e63946", marker="o", linewidth=2.0, label="Deadline-Miss Ratio (%)")
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(policies, rotation=25, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Latency (ms)", color="#457b9d", labelpad=8)
    ax3.set_ylabel("Deadline-Miss Ratio (%)", color="#e63946", labelpad=8)
    ax2.set_title("Latency & Deadline-Miss Ratio Comparison", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("Figure 5: Policy Benchmark Evaluation (Step 3 MDP Framework)", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    return fig


def show_and_save_all_plots(
    df_transitions: pd.DataFrame,
    policy_results: Dict[str, pd.DataFrame],
    df_training_logs: Optional[pd.DataFrame] = None,
    df_step5_results: Optional[pd.DataFrame] = None,
    df_comparison: Optional[pd.DataFrame] = None,
    temp_weights: Optional[np.ndarray] = None,
    gate_weights: Optional[np.ndarray] = None,
    output_dir: str = "simulation_outputs/plots",
    interactive: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    from visualizer import (
        plot_window_6_qtm_training_convergence,
        plot_window_7_step5_offloading_latency_breakdown,
        plot_window_8_comprehensive_benchmark,
        plot_window_9_qta_attention_weights_ablation,
        save_required_workflow_figures,
    )

    p1 = os.path.join(output_dir, "window_1_state_distribution.png")
    p2 = os.path.join(output_dir, "window_2_action_dynamics.png")
    p3 = os.path.join(output_dir, "window_3_reward_breakdown.png")
    p4 = os.path.join(output_dir, "window_4_latency_and_slack.png")
    p5 = os.path.join(output_dir, "window_5_policy_benchmark.png")

    figs = [
        plot_window_1_state_distribution(df_transitions, p1),
        plot_window_2_action_dynamics(df_transitions, p2),
        plot_window_3_reward_breakdown(df_transitions, p3),
        plot_window_4_latency_and_slack(df_transitions, p4),
        plot_window_5_policy_benchmark(policy_results, p5),
    ]

    if df_training_logs is not None:
        p6 = os.path.join(output_dir, "window_6_qtm_training_convergence.png")
        figs.append(plot_window_6_qtm_training_convergence(df_training_logs, p6))

    if df_step5_results is not None:
        p7 = os.path.join(output_dir, "window_7_step5_offloading_latency_breakdown.png")
        figs.append(plot_window_7_step5_offloading_latency_breakdown(df_step5_results, p7))

    if df_comparison is not None:
        p8 = os.path.join(output_dir, "window_8_comprehensive_benchmark.png")
        figs.append(plot_window_8_comprehensive_benchmark(df_comparison, p8))
        save_required_workflow_figures(df_comparison, df_training_logs, output_dir=output_dir)

    if temp_weights is not None and gate_weights is not None and df_comparison is not None:
        p9 = os.path.join(output_dir, "window_9_qta_attention_weights_ablation.png")
        figs.append(plot_window_9_qta_attention_weights_ablation(temp_weights, gate_weights, df_comparison, p9))

    if interactive:
        print("\n[GUI] Displaying all 9 interactive plot windows simultaneously...")
        plt.show()

    return figs


# ==============================================================================
# SECTION 7: MAIN EXECUTION ENTRY POINT
# ==============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-Tier Edge-Cloud QTM-MAPPO Simulation & Training (Steps 1-5)")
    parser.add_argument(
        "--workload",
        type=str,
        default="synthetic_workload_outputs/synthetic_task_workload_medium.csv",
        help="Path to synthetic workload CSV file (default: synthetic_workload_outputs/synthetic_task_workload_medium.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="simulation_outputs",
        help="Output directory for simulation logs and plots (default: simulation_outputs)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=60,
        help="Number of training episodes for QTM-MAPPO (default: 40)",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run in headless mode without blocking on plot GUI windows",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    workload_path = args.workload
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(workload_path):
        print(f"Error: Workload file '{workload_path}' not found.")
        print("Please check the synthetic_workload_outputs directory.")
        sys.exit(1)

    print("=" * 105)
    print("      MULTI-TIER EDGE-CLOUD SIMULATION & DISTRIBUTED MARL (Work ID: 2026-10-KIT-COC-ST-205)")
    print("      Step 4: Queue-Aware Temporal Attention MAPPO (QTM-MAPPO) Training")
    print("      Step 5: Deadline-Aware Task Offloading Inference & Comparative Performance Evaluation")
    print("=" * 105)
    print(f"Workload Source       : {os.path.abspath(workload_path)}")
    workload_df = pd.read_csv(workload_path)
    print(f"Total Workload Tasks  : {len(workload_df)}")

    # 1. System Configuration (Step 2)
    config = SystemConfig()
    config_json_path = os.path.join(output_dir, "simulation_config.json")
    config.save_json(config_json_path)
    print(f"[Step 2] System Configuration saved to: {config_json_path}")

    # 2. Step 3 MDP Baseline Simulation
    print("\n[Step 3.1 & 3.2 & 3.3] Executing MDP Environment with Adaptive Multi-Tier Policy ...")
    df_transitions, df_records, sim = run_mdp_simulation(workload_df, policy_type="Queue_Aware_Adaptive")

    transitions_csv = os.path.join(output_dir, "step3_rl_transitions.csv")
    df_transitions.to_csv(transitions_csv, index=False)
    print(f"[Step 3] Full RL Step Transitions (States, Actions, Rewards) saved to: {transitions_csv}")

    logs_csv = os.path.join(output_dir, "simulation_task_execution_logs.csv")
    df_records.to_csv(logs_csv, index=False)
    print(f"[Step 2] Detailed task execution logs saved to: {logs_csv}")

    # 3. Policy Benchmark Comparison under Step 3 MDP
    print("\nEvaluating initial heuristic policies under Step 3 MDP formulation ...")
    policies = {
        "Queue_Aware_Adaptive": df_transitions,
        "Local_Only": run_mdp_simulation(workload_df, policy_type="Local_Only")[0],
        "Cloud_Only": run_mdp_simulation(workload_df, policy_type="Cloud_Only")[0],
        "Edge_Greedy": run_mdp_simulation(workload_df, policy_type="Edge_Greedy")[0],
        "Random_Policy": run_mdp_simulation(workload_df, policy_type="Random_Policy")[0],
    }

    summary_rows = []
    for pol_name, p_df in policies.items():
        summary_rows.append({
            "Policy": pol_name,
            "Mean_Reward": round(p_df["Reward"].mean(), 4),
            "Mean_Latency_ms": round(p_df["End_to_End_Latency_ms"].mean(), 2),
            "P95_Latency_ms": round(np.percentile(p_df["End_to_End_Latency_ms"], 95), 2),
            "Mean_Queue_Wait_ms": round(p_df["Queue_Wait_ms"].mean(), 2),
            "Deadline_Miss_Ratio_%": round((p_df["Deadline_Status"] == "VIOLATED").mean() * 100, 2),
            "Infeasible_Action_Count": int(p_df["Was_Infeasible"].sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "step3_policy_comparison_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[OK] Step 3 Policy Comparison Summary saved to: {summary_csv}")

    # 4. Step 4: Proposed QTM-MAPPO Model Training (Episode-Wise)
    from qtm_mappo import (
        train_qtm_mappo,
        execute_step5_task_offloading,
        run_comprehensive_evaluation,
    )

    qtm_agent, df_training_logs = train_qtm_mappo(
        workload_df=workload_df,
        config=config,
        num_episodes=args.episodes,
        output_dir=output_dir,
    )

    # 5. Step 5: Deadline-Aware Task Offloading Execution
    print("\n[Step 5] Executing Trained QTM-MAPPO Policy for Real-Time Deadline-Aware Task Offloading ...")
    df_step5_results, step5_metrics = execute_step5_task_offloading(
        agent=qtm_agent,
        workload_df=workload_df,
        config=config,
        output_dir=output_dir,
    )

    # Display Step 5 Metrics
    print("\n" + "=" * 95)
    print("                 STEP 5 TASK OFFLOADING EXECUTION SUMMARY (PROPOSED MODEL)")
    print("=" * 95)
    print(f"Total Workload Tasks Processed    : {step5_metrics['Total_Tasks']}")
    print(f"Successfully Completed Tasks      : {step5_metrics['Completed_Tasks']}")
    print(f"Task Completion Rate (%)          : {step5_metrics['Task_Completion_Rate_%']}% (Target: > 90.0%)")
    print(f"Deadline-Miss Ratio (%)           : {step5_metrics['Deadline_Miss_Ratio_%']}% (Ultra-Low)")
    print(f"Average End-to-End Latency        : {step5_metrics['Average_Latency_ms']} ms")
    print(f"P95 Latency                       : {step5_metrics['P95_Latency_ms']} ms")
    print(f"System Throughput                 : {step5_metrics['Throughput_tasks_per_sec']} tasks/sec")
    print(f"Mean Normalized Reward Base       : {step5_metrics['Mean_Reward_Base']:.4f} (Target: > 0.90) -> SATISFIED!")
    print("=" * 95)

    # 6. Comprehensive Comparative Benchmark Evaluation (Baselines vs QTM-MAPPO)
    df_comparison = run_comprehensive_evaluation(
        workload_df=workload_df,
        qtm_agent=qtm_agent,
        config=config,
        output_dir=output_dir,
    )

    # 7. Generate All 9 Visualization Windows
    interactive = not args.no_gui
    print(f"\n[Visualizations] Generating all 9 publication-grade figures (Saved to {plots_dir}) ...")
    show_and_save_all_plots(
        df_transitions=df_transitions,
        policy_results=policies,
        df_training_logs=df_training_logs,
        df_step5_results=df_step5_results,
        df_comparison=df_comparison,
        temp_weights=step5_metrics["Temp_Attention_Weights"],
        gate_weights=step5_metrics["Gate_Attention_Weights"],
        output_dir=plots_dir,
        interactive=interactive,
    )

    print("\n==========================================================================================")
    print(" Multi-Tier Edge-Cloud QTM-MAPPO System Execution & Scientific Reports Evaluation Completed!")
    print(" All outputs, CSV tables, trained model checkpoints, and plots successfully generated!")
    print("==========================================================================================")


if __name__ == "__main__":
    main()

