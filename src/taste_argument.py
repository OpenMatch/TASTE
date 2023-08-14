from dataclasses import dataclass, field


@dataclass
class TASTEArguments:
    num_passages: int = field(
        default=2,
        metadata={"help": "the num of sequence text split chunk"}
    )
