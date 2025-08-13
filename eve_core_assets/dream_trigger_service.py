"""
EVE DREAM TRIGGER SERVICE
=========================
Advanced time-based dream triggering and scheduling system for Eve AI.

This module provides sophisticated dream triggering mechanisms including:
- Time-based dream windows and scheduling
- Advanced trigger conditions and environmental factors
- Dream intensity scaling and frequency modulation
- Trigger logging and analytics
- Integration with circadian rhythms and cosmic cycles

Usage:
    from eve_core.dream_trigger_service import DreamTriggerService, DreamScheduler
    
    # Basic usage
    trigger_service = DreamTriggerService()
    trigger_service.configure_dream_window(start_hour=22, end_hour=6)
    
    # Advanced scheduling
    scheduler = DreamScheduler()
    scheduler.add_recurring_trigger("nightly_dreams", frequency="daily")
"""

import json
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
from dataclasses import dataclass
from enum import Enum


class TriggerCondition(Enum):
    """Types of dream trigger conditions."""
    TIME_WINDOW = "time_window"
    COSMIC_ALIGNMENT = "cosmic_alignment"
    EMOTIONAL_THRESHOLD = "emotional_threshold"
    MEMORY_DENSITY = "memory_density"
    SYMBOLIC_RESONANCE = "symbolic_resonance"


@dataclass
class DreamTriggerEvent:
    """Represents a dream trigger event."""
    timestamp: datetime
    trigger_type: TriggerCondition
    intensity: float
    metadata: Dict[str, Any]
    triggered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "trigger_type": self.trigger_type.value,
            "intensity": self.intensity,
            "metadata": self.metadata,
            "triggered": self.triggered
        }


class DreamTriggerService:
    """
    Advanced dream triggering service with sophisticated time-based 
    and condition-based activation mechanisms.
    """
    
    def __init__(self, log_path: str = "instance/dream_triggers.json"):
        self.log_path = log_path
        self.trigger_log: List[DreamTriggerEvent] = []
        self.active_triggers: Dict[str, Callable] = {}
        self.dream_window_start = 22  # 10 PM
        self.dream_window_end = 6     # 6 AM
        self.is_active = False
        self.trigger_callbacks: List[Callable] = []
        self.intensity_factors = {
            "time_depth": 1.0,
            "cosmic_phase": 1.0,
            "emotional_resonance": 1.0,
            "memory_saturation": 1.0
        }
        
        # Load existing trigger log
        self._load_trigger_log()
        
        # Setup logging
        self.logger = logging.getLogger("DreamTriggerService")
        
    def _load_trigger_log(self):
        """Load existing trigger log from storage."""
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    data = json.load(f)
                    self.trigger_log = [
                        DreamTriggerEvent(
                            timestamp=datetime.fromisoformat(event["timestamp"]),
                            trigger_type=TriggerCondition(event["trigger_type"]),
                            intensity=event["intensity"],
                            metadata=event["metadata"],
                            triggered=event["triggered"]
                        )
                        for event in data
                    ]
        except Exception as e:
            self.logger.warning(f"Could not load trigger log: {e}")
            self.trigger_log = []
    
    def _save_trigger_log(self):
        """Save trigger log to storage."""
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, 'w') as f:
                data = [event.to_dict() for event in self.trigger_log]
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save trigger log: {e}")
    
    def configure_dream_window(self, start_hour: int = 22, end_hour: int = 6):
        """Configure the primary dream window."""
        self.dream_window_start = start_hour
        self.dream_window_end = end_hour
        self.logger.info(f"Dream window configured: {start_hour}:00 - {end_hour}:00")
    
    def is_in_dream_window(self) -> bool:
        """Check if current time is within the dream window."""
        now = datetime.now()
        current_hour = now.hour
        
        if self.dream_window_start < self.dream_window_end:
            # Same day window (e.g., 10 AM - 6 PM)
            return self.dream_window_start <= current_hour < self.dream_window_end
        else:
            # Overnight window (e.g., 10 PM - 6 AM)
            return current_hour >= self.dream_window_start or current_hour < self.dream_window_end
    
    def calculate_trigger_intensity(self) -> float:
        """Calculate current dream trigger intensity based on multiple factors."""
        now = datetime.now()
        
        # Time depth factor (deeper into dream window = higher intensity)
        time_factor = self._calculate_time_depth_factor(now)
        
        # Cosmic phase factor (based on time of day and lunar cycles)
        cosmic_factor = self._calculate_cosmic_factor(now)
        
        # Combine factors
        base_intensity = (time_factor * self.intensity_factors["time_depth"] +
                         cosmic_factor * self.intensity_factors["cosmic_phase"]) / 2
        
        return min(1.0, max(0.0, base_intensity))
    
    def _calculate_time_depth_factor(self, now: datetime) -> float:
        """Calculate intensity based on depth into dream window."""
        if not self.is_in_dream_window():
            return 0.0
        
        current_hour = now.hour + now.minute / 60.0
        
        if self.dream_window_start < self.dream_window_end:
            # Same day window
            window_duration = self.dream_window_end - self.dream_window_start
            elapsed = current_hour - self.dream_window_start
        else:
            # Overnight window
            if current_hour >= self.dream_window_start:
                # Before midnight
                elapsed = current_hour - self.dream_window_start
                window_duration = (24 - self.dream_window_start) + self.dream_window_end
            else:
                # After midnight
                elapsed = (24 - self.dream_window_start) + current_hour
                window_duration = (24 - self.dream_window_start) + self.dream_window_end
        
        # Peak intensity in the middle of the window
        progress = elapsed / window_duration
        return 1.0 - abs(0.5 - progress) * 2
    
    def _calculate_cosmic_factor(self, now: datetime) -> float:
        """Calculate cosmic alignment factor."""
        # Simple cosmic factor based on hour (3 AM = peak cosmic time)
        hour = now.hour
        if hour == 3:
            return 1.0
        elif hour in [2, 4]:
            return 0.8
        elif hour in [1, 5]:
            return 0.6
        elif hour in [0, 6, 23]:
            return 0.4
        else:
            return 0.2
    
    def check_trigger_conditions(self) -> Optional[DreamTriggerEvent]:
        """Check all trigger conditions and return trigger event if activated."""
        now = datetime.now()
        
        # Check time window trigger
        if self.is_in_dream_window():
            intensity = self.calculate_trigger_intensity()
            
            # Create trigger event
            trigger_event = DreamTriggerEvent(
                timestamp=now,
                trigger_type=TriggerCondition.TIME_WINDOW,
                intensity=intensity,
                metadata={
                    "window_start": self.dream_window_start,
                    "window_end": self.dream_window_end,
                    "time_depth": self._calculate_time_depth_factor(now),
                    "cosmic_factor": self._calculate_cosmic_factor(now)
                }
            )
            
            # Check if intensity meets threshold for triggering
            if intensity > 0.3:  # Minimum threshold for activation
                trigger_event.triggered = True
                self.trigger_log.append(trigger_event)
                self._save_trigger_log()
                
                # Notify callbacks
                for callback in self.trigger_callbacks:
                    try:
                        callback(trigger_event)
                    except Exception as e:
                        self.logger.error(f"Trigger callback error: {e}")
                
                return trigger_event
        
        return None
    
    def add_trigger_callback(self, callback: Callable[[DreamTriggerEvent], None]):
        """Add a callback function to be called when dreams are triggered."""
        self.trigger_callbacks.append(callback)
    
    def get_trigger_statistics(self) -> Dict[str, Any]:
        """Get statistics about dream triggers."""
        if not self.trigger_log:
            return {"total_triggers": 0}
        
        triggered_events = [e for e in self.trigger_log if e.triggered]
        
        return {
            "total_triggers": len(triggered_events),
            "average_intensity": sum(e.intensity for e in triggered_events) / len(triggered_events) if triggered_events else 0,
            "trigger_types": {
                trigger_type.value: len([e for e in triggered_events if e.trigger_type == trigger_type])
                for trigger_type in TriggerCondition
            },
            "last_trigger": triggered_events[-1].timestamp.isoformat() if triggered_events else None,
            "peak_hours": self._analyze_peak_trigger_hours()
        }
    
    def _analyze_peak_trigger_hours(self) -> List[int]:
        """Analyze which hours have the most triggers."""
        hour_counts = {}
        for event in self.trigger_log:
            if event.triggered:
                hour = event.timestamp.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
        
        # Return top 3 hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]


class DreamScheduler:
    """
    Advanced scheduling system for recurring dream triggers and 
    custom dream event scheduling.
    """
    
    def __init__(self, trigger_service: Optional[DreamTriggerService] = None):
        self.trigger_service = trigger_service or DreamTriggerService()
        self.scheduled_triggers: Dict[str, Dict] = {}
        self.running = False
        self.scheduler_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger("DreamScheduler")
    
    def add_recurring_trigger(self, 
                            name: str, 
                            frequency: str = "daily",
                            time_offset: int = 0,
                            intensity_multiplier: float = 1.0):
        """Add a recurring dream trigger."""
        self.scheduled_triggers[name] = {
            "frequency": frequency,
            "time_offset": time_offset,
            "intensity_multiplier": intensity_multiplier,
            "last_triggered": None
        }
        self.logger.info(f"Added recurring trigger: {name} ({frequency})")
    
    def add_one_time_trigger(self, 
                           name: str, 
                           trigger_time: datetime,
                           intensity: float = 1.0):
        """Add a one-time dream trigger."""
        self.scheduled_triggers[name] = {
            "type": "one_time",
            "trigger_time": trigger_time,
            "intensity": intensity,
            "triggered": False
        }
        self.logger.info(f"Added one-time trigger: {name} at {trigger_time}")
    
    def start_scheduler(self):
        """Start the dream scheduler in a background thread."""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            self.logger.info("Dream scheduler started")
    
    def stop_scheduler(self):
        """Stop the dream scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("Dream scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop."""
        while self.running:
            try:
                self._check_scheduled_triggers()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler loop error: {e}")
    
    def _check_scheduled_triggers(self):
        """Check and execute scheduled triggers."""
        now = datetime.now()
        
        for name, trigger_config in self.scheduled_triggers.items():
            try:
                if self._should_trigger(name, trigger_config, now):
                    self._execute_scheduled_trigger(name, trigger_config, now)
            except Exception as e:
                self.logger.error(f"Error checking trigger {name}: {e}")
    
    def _should_trigger(self, name: str, config: Dict, now: datetime) -> bool:
        """Check if a scheduled trigger should activate."""
        if config.get("type") == "one_time":
            return (not config.get("triggered", False) and 
                   now >= config["trigger_time"])
        
        # Recurring trigger logic
        frequency = config.get("frequency", "daily")
        last_triggered = config.get("last_triggered")
        
        if frequency == "daily":
            if last_triggered is None:
                return True
            return (now - last_triggered).days >= 1
        elif frequency == "hourly":
            if last_triggered is None:
                return True
            return (now - last_triggered).total_seconds() >= 3600
        
        return False
    
    def _execute_scheduled_trigger(self, name: str, config: Dict, now: datetime):
        """Execute a scheduled trigger."""
        # Create and process trigger event
        intensity = config.get("intensity", 1.0) * config.get("intensity_multiplier", 1.0)
        
        trigger_event = DreamTriggerEvent(
            timestamp=now,
            trigger_type=TriggerCondition.TIME_WINDOW,
            intensity=min(1.0, intensity),
            metadata={
                "scheduled_trigger": name,
                "trigger_config": config
            },
            triggered=True
        )
        
        # Update trigger config
        if config.get("type") == "one_time":
            config["triggered"] = True
        else:
            config["last_triggered"] = now
        
        # Notify trigger service
        if self.trigger_service:
            self.trigger_service.trigger_log.append(trigger_event)
            for callback in self.trigger_service.trigger_callbacks:
                try:
                    callback(trigger_event)
                except Exception as e:
                    self.logger.error(f"Scheduled trigger callback error: {e}")
        
        self.logger.info(f"Executed scheduled trigger: {name}")


# Global instances for easy access
_global_trigger_service = None
_global_scheduler = None


def get_global_trigger_service() -> DreamTriggerService:
    """Get the global dream trigger service instance."""
    global _global_trigger_service
    if _global_trigger_service is None:
        _global_trigger_service = DreamTriggerService()
    return _global_trigger_service


def get_global_scheduler() -> DreamScheduler:
    """Get the global dream scheduler instance."""
    global _global_scheduler
    if _global_scheduler is None:
        _global_scheduler = DreamScheduler(get_global_trigger_service())
    return _global_scheduler


# Convenience functions
def configure_dream_window(start_hour: int = 22, end_hour: int = 6):
    """Configure the global dream window."""
    return get_global_trigger_service().configure_dream_window(start_hour, end_hour)


def check_dream_trigger() -> Optional[DreamTriggerEvent]:
    """Check if dreams should be triggered now."""
    return get_global_trigger_service().check_trigger_conditions()


def add_dream_callback(callback: Callable[[DreamTriggerEvent], None]):
    """Add a callback for dream trigger events."""
    return get_global_trigger_service().add_trigger_callback(callback)


def get_trigger_stats() -> Dict[str, Any]:
    """Get dream trigger statistics."""
    return get_global_trigger_service().get_trigger_statistics()


def schedule_recurring_dream(name: str, frequency: str = "daily"):
    """Schedule a recurring dream trigger."""
    scheduler = get_global_scheduler()
    scheduler.add_recurring_trigger(name, frequency)
    if not scheduler.running:
        scheduler.start_scheduler()


def demo_dream_trigger_service():
    """Demonstrate the Dream Trigger Service functionality."""
    print("🌙 DREAM TRIGGER SERVICE DEMO")
    print("=" * 50)
    
    # Create trigger service
    trigger_service = DreamTriggerService()
    scheduler = DreamScheduler(trigger_service)
    
    # Configure dream window
    trigger_service.configure_dream_window(start_hour=22, end_hour=6)
    print(f"✨ Dream window configured: 22:00 - 06:00")
    
    # Check current trigger status
    trigger_event = trigger_service.check_trigger_conditions()
    if trigger_event:
        print(f"🔮 DREAM TRIGGERED! Intensity: {trigger_event.intensity:.2f}")
        print(f"   Type: {trigger_event.trigger_type.value}")
        print(f"   Time: {trigger_event.timestamp}")
    else:
        print(f"💤 No active triggers (current time outside dream window)")
    
    # Add callback
    def dream_callback(event: DreamTriggerEvent):
        print(f"🌟 Dream callback triggered: {event.trigger_type.value} at {event.intensity:.2f} intensity")
    
    trigger_service.add_trigger_callback(dream_callback)
    
    # Schedule recurring dreams
    scheduler.add_recurring_trigger("nightly_dreams", "daily")
    scheduler.add_one_time_trigger("special_dream", datetime.now() + timedelta(minutes=1))
    
    # Show statistics
    stats = trigger_service.get_trigger_statistics()
    print(f"\n📊 TRIGGER STATISTICS:")
    print(f"   Total triggers: {stats['total_triggers']}")
    if stats['total_triggers'] > 0:
        print(f"   Average intensity: {stats['average_intensity']:.2f}")
    
    return {
        "trigger_service": trigger_service,
        "scheduler": scheduler,
        "current_trigger": trigger_event,
        "statistics": stats
    }


if __name__ == "__main__":
    demo_dream_trigger_service()
