"""Startup/end membership witness; ordinary observation hooks remain opt-in."""
import json
import os
from pathlib import Path

from observation_worker import ObservationWorker
from capacity_contract import membership_hash


class CapacityWorker(ObservationWorker):
    def capacity_identity(self, folder, phase, group_incarnation):
        if phase not in ('loaded','after'):raise ValueError('Unsupported identity phase')
        import torch
        from vllm.distributed import get_ep_group
        group=get_ep_group();world=group.world_size
        if world not in (1,2):raise ValueError('Only EP1 or EP2 is allowed, never EP4')
        rank=self.parallel_config.data_parallel_rank
        gpu=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        value=membership_hash(group_incarnation,gpu,rank)
        # Tiny outside-window identity collective, not an inference or bandwidth test.
        if world==1:members=[value]
        else:
            tensor=torch.tensor(value,dtype=torch.int32,device='cuda')
            members=group.all_gather(tensor,dim=0).reshape(world,32).cpu().tolist()
        result=dict(pid=os.getpid(),rank=rank,gpu_uuid=gpu,group_incarnation=group_incarnation,members=members,
                    provenance='Configured group nonce + observed GPU UUIDs participated in this EP collective; not cryptographic attestation')
        path=Path(folder)/f'identity-{phase}-dp{rank}.json'
        with path.open('x') as f:json.dump(result,f,indent=2)
        return dict(rank=rank,written=str(path))
