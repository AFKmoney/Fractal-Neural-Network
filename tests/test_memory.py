import pytest
import torch
from nfn.memory import WorkingMemory

def test_save_state_none():
    # Initialize WorkingMemory
    memory = WorkingMemory(d_model=16, n_slots=8, n_heads=4)
    # Ensure initial state is None
    assert memory._state is None
    # Save state should return None
    saved_state = memory.save_state()
    assert saved_state is None

def test_save_state_detached_copy():
    # Initialize WorkingMemory
    memory = WorkingMemory(d_model=16, n_slots=8, n_heads=4)
    batch_size = 2
    device = torch.device("cpu")

    # Initialize state
    memory.reset(batch_size, device)

    assert memory._state is not None
    original_state = memory._state.clone()

    # Save state
    saved_state = memory.save_state()

    # Should be equal to the current state
    assert torch.equal(saved_state, memory._state)

    # But it shouldn't be the same object in memory
    assert saved_state is not memory._state

    # Modifying the saved state shouldn't affect the memory state
    saved_state[0, 0, 0] = 999.0
    assert not torch.equal(saved_state, memory._state)
    assert torch.equal(memory._state, original_state)

    # Should not have gradients or require grad (it's detached)
    assert not saved_state.requires_grad

def test_get_state_initializes_when_none():
    memory = WorkingMemory(d_model=16, n_slots=8, n_heads=4)
    batch_size = 2
    device = torch.device("cpu")

    assert memory._state is None
    state = memory.get_state(batch_size, device)

    # State should now be initialized
    assert memory._state is not None
    assert state.shape == (batch_size, 8, 16)
    assert state.device == device

def test_get_state_reinitializes_when_batch_size_changes():
    memory = WorkingMemory(d_model=16, n_slots=8, n_heads=4)
    device = torch.device("cpu")

    # Initialized with batch size 2
    memory.reset(2, device)
    assert memory._state.shape[0] == 2

    # Request state for batch size 4
    state = memory.get_state(4, device)

    # State should be reinitialized with batch size 4
    assert memory._state.shape[0] == 4
    assert state.shape[0] == 4

def test_set_state():
    memory = WorkingMemory(d_model=16, n_slots=8, n_heads=4)

    new_state = torch.ones(2, 8, 16, requires_grad=True)
    memory.set_state(new_state)

    # Memory state should be the new state
    assert torch.equal(memory._state, torch.ones(2, 8, 16))
    # It should be detached
    assert not memory._state.requires_grad
