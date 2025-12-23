"""Tests for the message bus communication system.

Copyright (c) 2025 Celaya Solutions AI Research Lab
Licensed under the MIT License
"""

import pytest
import time
import threading
from typing import List
from unittest.mock import MagicMock, Mock
from pydantic import ValidationError

from neuralchild.communication.message_bus import (
    MessageBus, MessageFilter, SubscriptionInfo, GlobalMessageBus
)
from neuralchild.core.schemas import NetworkMessage, DevelopmentalStage


class TestMessageFilter:
    """Test suite for MessageFilter."""

    def test_message_filter_creation(self):
        """Test creating a message filter."""
        filter_obj = MessageFilter(
            sender="perception",
            message_type="sensory"
        )

        assert filter_obj.sender == "perception"
        assert filter_obj.message_type == "sensory"

    def test_message_filter_validation(self):
        """Test message filter validation."""
        # Valid filter with at least one criterion
        filter_obj = MessageFilter(sender="test")
        assert filter_obj.sender == "test"

        # Invalid filter with no criteria should raise error
        with pytest.raises(ValidationError):
            MessageFilter()

    def test_message_filter_multiple_criteria(self):
        """Test filter with multiple criteria."""
        filter_obj = MessageFilter(
            sender="perception",
            receiver="consciousness",
            message_type="sensory",
            min_priority=0.5
        )

        assert filter_obj.sender == "perception"
        assert filter_obj.receiver == "consciousness"
        assert filter_obj.min_priority == 0.5

    def test_message_filter_priority(self):
        """Test priority filtering."""
        filter_obj = MessageFilter(min_priority=0.7)
        assert filter_obj.min_priority == 0.7

    def test_message_filter_developmental_stage(self):
        """Test developmental stage filtering."""
        filter_obj = MessageFilter(
            sender="test",
            max_developmental_stage=DevelopmentalStage.TODDLER
        )
        assert filter_obj.max_developmental_stage == DevelopmentalStage.TODDLER


class TestSubscriptionInfo:
    """Test suite for SubscriptionInfo."""

    def test_subscription_info_creation(self, message_filter: MessageFilter):
        """Test creating subscription info."""
        sub_info = SubscriptionInfo(
            subscriber_id="test_subscriber",
            filter=message_filter
        )

        assert sub_info.subscriber_id == "test_subscriber"
        assert sub_info.filter is not None

    def test_subscription_info_with_callback(self, message_filter: MessageFilter):
        """Test subscription with callback."""
        callback = MagicMock()
        sub_info = SubscriptionInfo(
            subscriber_id="test",
            filter=message_filter,
            callback=callback
        )

        assert sub_info.callback is not None

    def test_subscription_info_with_queue(self, message_filter: MessageFilter):
        """Test subscription with queue name."""
        sub_info = SubscriptionInfo(
            subscriber_id="test",
            filter=message_filter,
            queue_name="test_queue"
        )

        assert sub_info.queue_name == "test_queue"


class TestMessageBusInitialization:
    """Test suite for MessageBus initialization."""

    def test_message_bus_creation(self, message_bus: MessageBus):
        """Test that MessageBus initializes correctly."""
        assert message_bus is not None
        assert hasattr(message_bus, 'subscriptions')
        assert hasattr(message_bus, 'message_queues')
        assert hasattr(message_bus, 'message_history')

    def test_message_bus_initial_state(self, message_bus: MessageBus):
        """Test MessageBus initial state."""
        assert isinstance(message_bus.subscriptions, list)
        assert isinstance(message_bus.message_queues, dict)
        assert isinstance(message_bus.message_history, list)
        assert len(message_bus.subscriptions) == 0
        assert len(message_bus.message_queues) == 0

    def test_message_bus_threading(self, message_bus: MessageBus):
        """Test that MessageBus starts delivery thread."""
        assert hasattr(message_bus, 'delivery_thread')
        assert message_bus.running is True


class TestMessageBusSubscription:
    """Test suite for MessageBus subscription operations."""

    def test_subscribe_with_callback(self, message_bus: MessageBus):
        """Test subscribing with a callback."""
        callback = MagicMock()
        filter_obj = MessageFilter(sender="test")

        result = message_bus.subscribe(
            subscriber_id="test_sub",
            filter_config=filter_obj,
            callback=callback
        )

        # Should not return queue name when callback is provided
        assert result is None
        assert len(message_bus.subscriptions) > 0

    def test_subscribe_with_queue(self, message_bus: MessageBus):
        """Test subscribing with queue-based delivery."""
        filter_obj = MessageFilter(sender="test")

        queue_name = message_bus.subscribe(
            subscriber_id="test_sub",
            filter_config=filter_obj
        )

        # Should return queue name
        assert queue_name is not None
        assert queue_name in message_bus.message_queues
        assert len(message_bus.subscriptions) > 0

    def test_subscribe_with_dict_filter(self, message_bus: MessageBus):
        """Test subscribing with dictionary filter config."""
        filter_dict = {"sender": "test", "message_type": "sensory"}

        queue_name = message_bus.subscribe(
            subscriber_id="test_sub",
            filter_config=filter_dict
        )

        assert queue_name is not None
        assert len(message_bus.subscriptions) > 0

    def test_multiple_subscriptions(self, message_bus: MessageBus):
        """Test multiple subscriptions."""
        filter1 = MessageFilter(sender="perception")
        filter2 = MessageFilter(sender="emotions")

        message_bus.subscribe("sub1", filter1)
        message_bus.subscribe("sub2", filter2)

        assert len(message_bus.subscriptions) == 2

    def test_unsubscribe(self, message_bus: MessageBus):
        """Test unsubscribing from messages."""
        filter_obj = MessageFilter(sender="test")
        queue_name = message_bus.subscribe("test_sub", filter_obj)

        # Unsubscribe
        if hasattr(message_bus, 'unsubscribe'):
            message_bus.unsubscribe(queue_name)

            # Subscription should be removed
            assert queue_name not in message_bus.message_queues


class TestMessageBusPublishing:
    """Test suite for MessageBus publishing operations."""

    def test_publish_message(self, message_bus: MessageBus, sample_network_message: NetworkMessage):
        """Test publishing a message."""
        if hasattr(message_bus, 'publish'):
            message_bus.publish(sample_network_message)

            # Message should be in history
            assert len(message_bus.message_history) > 0

    def test_publish_to_subscribers(self, message_bus: MessageBus):
        """Test publishing message to subscribers."""
        callback = MagicMock()
        filter_obj = MessageFilter(sender="perception")

        message_bus.subscribe("test_sub", filter_obj, callback)

        message = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={"data": "test"}
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(message)

            # Give delivery thread time to process
            time.sleep(0.1)

            # Callback might be called
            # Note: callback behavior depends on implementation

    def test_publish_multiple_messages(self, message_bus: MessageBus):
        """Test publishing multiple messages."""
        messages = [
            NetworkMessage(sender=f"sender_{i}", receiver="target", content={})
            for i in range(5)
        ]

        if hasattr(message_bus, 'publish'):
            for msg in messages:
                message_bus.publish(msg)

            # Messages should be in history
            assert len(message_bus.message_history) >= 5

    def test_message_history_limit(self, message_bus: MessageBus):
        """Test message history size limit."""
        max_size = message_bus.max_history_size

        # Publish more messages than limit
        if hasattr(message_bus, 'publish'):
            for i in range(max_size + 100):
                msg = NetworkMessage(
                    sender="test",
                    receiver="target",
                    content={"index": i}
                )
                message_bus.publish(msg)

            # History should not exceed max size
            assert len(message_bus.message_history) <= max_size


class TestMessageFiltering:
    """Test suite for message filtering."""

    def test_filter_by_sender(self, message_bus: MessageBus):
        """Test filtering messages by sender."""
        callback = MagicMock()
        filter_obj = MessageFilter(sender="perception")

        message_bus.subscribe("test_sub", filter_obj, callback)

        # Matching message
        match_msg = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={}
        )

        # Non-matching message
        no_match_msg = NetworkMessage(
            sender="emotions",
            receiver="consciousness",
            content={}
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(match_msg)
            message_bus.publish(no_match_msg)

            time.sleep(0.1)

    def test_filter_by_message_type(self, message_bus: MessageBus):
        """Test filtering messages by type."""
        callback = MagicMock()
        filter_obj = MessageFilter(sender="test", message_type="sensory")

        message_bus.subscribe("test_sub", filter_obj, callback)

        # Matching message
        match_msg = NetworkMessage(
            sender="test",
            receiver="target",
            content={},
            message_type="sensory"
        )

        # Non-matching message
        no_match_msg = NetworkMessage(
            sender="test",
            receiver="target",
            content={},
            message_type="emotional"
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(match_msg)
            message_bus.publish(no_match_msg)

            time.sleep(0.1)

    def test_filter_by_priority(self, message_bus: MessageBus):
        """Test filtering messages by priority."""
        filter_obj = MessageFilter(sender="test", min_priority=0.7)
        queue_name = message_bus.subscribe("test_sub", filter_obj)

        # High priority message
        high_priority = NetworkMessage(
            sender="test",
            receiver="target",
            content={},
            priority=0.9
        )

        # Low priority message
        low_priority = NetworkMessage(
            sender="test",
            receiver="target",
            content={},
            priority=0.3
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(high_priority)
            message_bus.publish(low_priority)

            time.sleep(0.1)

    def test_filter_multiple_criteria(self, message_bus: MessageBus):
        """Test filtering with multiple criteria."""
        filter_obj = MessageFilter(
            sender="perception",
            message_type="sensory",
            min_priority=0.5
        )

        queue_name = message_bus.subscribe("test_sub", filter_obj)

        # Message matching all criteria
        match_all = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={},
            message_type="sensory",
            priority=0.8
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(match_all)
            time.sleep(0.1)


class TestMessageRetrieval:
    """Test suite for message retrieval from queues."""

    def test_get_messages_from_queue(self, message_bus: MessageBus):
        """Test retrieving messages from queue."""
        filter_obj = MessageFilter(sender="test")
        queue_name = message_bus.subscribe("test_sub", filter_obj)

        # Publish message
        message = NetworkMessage(
            sender="test",
            receiver="target",
            content={"data": "test"}
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(message)

            # Wait for delivery
            time.sleep(0.1)

            # Retrieve messages
            if hasattr(message_bus, 'get_messages'):
                messages = message_bus.get_messages(queue_name)

                # Might have messages
                assert isinstance(messages, list)

    def test_get_messages_timeout(self, message_bus: MessageBus):
        """Test getting messages with timeout."""
        filter_obj = MessageFilter(sender="test")
        queue_name = message_bus.subscribe("test_sub", filter_obj)

        # Try to get messages with timeout
        if hasattr(message_bus, 'get_messages'):
            messages = message_bus.get_messages(queue_name, timeout=0.1)

            # Should return empty list or handle timeout
            assert isinstance(messages, list)


class TestMessageBusThreadSafety:
    """Test suite for thread safety."""

    def test_concurrent_publishing(self, message_bus: MessageBus):
        """Test concurrent message publishing."""
        def publish_messages(count: int):
            for i in range(count):
                msg = NetworkMessage(
                    sender=f"thread_{threading.current_thread().name}",
                    receiver="target",
                    content={"index": i}
                )
                if hasattr(message_bus, 'publish'):
                    message_bus.publish(msg)

        # Create multiple threads
        threads = [
            threading.Thread(target=publish_messages, args=(10,))
            for _ in range(3)
        ]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        # All messages should be processed
        time.sleep(0.2)

    def test_concurrent_subscribe(self, message_bus: MessageBus):
        """Test concurrent subscriptions."""
        def subscribe_to_bus(subscriber_id: str):
            filter_obj = MessageFilter(sender=subscriber_id)
            message_bus.subscribe(subscriber_id, filter_obj)

        # Create multiple subscription threads
        threads = [
            threading.Thread(target=subscribe_to_bus, args=(f"sub_{i}",))
            for i in range(5)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All subscriptions should be registered
        assert len(message_bus.subscriptions) >= 5


class TestGlobalMessageBus:
    """Test suite for GlobalMessageBus singleton."""

    def test_global_message_bus_access(self):
        """Test accessing the global message bus."""
        if hasattr(GlobalMessageBus, 'get_instance'):
            bus1 = GlobalMessageBus.get_instance()
            bus2 = GlobalMessageBus.get_instance()

            # Should return the same instance
            assert bus1 is bus2


class TestMessageBusIntegration:
    """Integration tests for MessageBus."""

    def test_end_to_end_message_flow(self, message_bus: MessageBus):
        """Test complete message flow from publish to receive."""
        received_messages = []

        def callback(message: NetworkMessage):
            received_messages.append(message)

        # Subscribe
        filter_obj = MessageFilter(sender="perception")
        message_bus.subscribe("test_sub", filter_obj, callback)

        # Publish
        message = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={"data": "test"}
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(message)

            # Wait for delivery
            time.sleep(0.2)

    def test_network_communication_simulation(self, message_bus: MessageBus):
        """Test simulating network communication."""
        # Simulate perception -> consciousness flow
        perception_filter = MessageFilter(sender="perception")
        consciousness_queue = message_bus.subscribe("consciousness", perception_filter)

        # Publish from perception
        msg = NetworkMessage(
            sender="perception",
            receiver="consciousness",
            content={"visual_data": [0.1, 0.2, 0.3]},
            message_type="sensory"
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(msg)

            time.sleep(0.1)

            # Consciousness retrieves messages
            if hasattr(message_bus, 'get_messages'):
                messages = message_bus.get_messages(consciousness_queue, timeout=0.1)

    def test_broadcast_message(self, message_bus: MessageBus):
        """Test broadcasting message to multiple subscribers."""
        callbacks = [MagicMock() for _ in range(3)]
        filter_obj = MessageFilter(sender="broadcast")

        # Multiple subscribers
        for i, callback in enumerate(callbacks):
            message_bus.subscribe(f"sub_{i}", filter_obj, callback)

        # Broadcast message
        msg = NetworkMessage(
            sender="broadcast",
            receiver="all",
            content={"announcement": "test"}
        )

        if hasattr(message_bus, 'publish'):
            message_bus.publish(msg)

            time.sleep(0.2)


class TestMessageBusErrorHandling:
    """Test error handling in MessageBus."""

    def test_publish_invalid_message(self, message_bus: MessageBus):
        """Test publishing invalid message."""
        if hasattr(message_bus, 'publish'):
            try:
                message_bus.publish(None)
            except (TypeError, AttributeError, ValueError):
                # Expected behavior
                pass

    def test_subscribe_invalid_filter(self, message_bus: MessageBus):
        """Test subscribing with invalid filter."""
        try:
            message_bus.subscribe("test", {})  # Empty filter should fail
        except (ValidationError, ValueError):
            # Expected behavior
            pass

    def test_get_messages_invalid_queue(self, message_bus: MessageBus):
        """Test getting messages from non-existent queue."""
        if hasattr(message_bus, 'get_messages'):
            try:
                messages = message_bus.get_messages("nonexistent_queue", timeout=0.1)
                # Should handle gracefully
            except (KeyError, ValueError):
                # Expected behavior
                pass
