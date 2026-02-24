from econsimulacra.animations import GridSpaceAnimator
from econsimulacra.animations import SocialNetworkAnimator
# export CONFIG_PATH="config.json"
# export LOG_TXT_PATH="log.txt"


class SimpleGridAnimator(GridSpaceAnimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class SimpleSocialAnimator(SocialNetworkAnimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
