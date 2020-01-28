# -*- coding: utf-8 -*-
"""
動いて、壁に当たると反射するアレの座標を計算する
===============================================

ザックリ設計資料は以下を参照。
https://github.com/toru-ver4/sample_code/issues/50

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries

# information
__author__ = 'Toru Yoshihara'
__copyright__ = 'Copyright (C) 2019 - Toru Yoshihara'
__license__ = 'New BSD License - https://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Toru Yoshihara'
__email__ = 'toru.ver.11 at-sign gmail.com'

__all__ = []


class ReflectiveMovingObject:
    """
    動いて、壁に当たると反射するアレの座標を計算するクラス
    """
    def __init__(self, pos_init=(0, 0), velocity_init=(5, 5), radius=30,
                 outline_size=(1920, 1080)):
        """
        pos や outline_width, outline_height の座標系は
        outline_size から radius * 2 を引いた座標系であることに注意。
        計算を楽にするためにこうしている。
        なお、後述の get_pos() でユーザーの使用する座標系に変換する。
        """
        self.radius = radius
        self.outline_width = outline_size[0] - self.radius * 2
        self.outline_height = outline_size[1] - self.radius * 2
        self.pos = [pos_init[0], pos_init[1]]
        self.velocity = [0, 0]
        self.set_velocity(velocity_init)

    def set_velocity(self, velocity=(5, 5)):
        """
        velocityの更新。
        あまりにも移動量が多いと色々とおかしくなるので、
        最大値には制限をかける。
        """
        if (abs(velocity[0]) > self.outline_width // 2):
            self.velocity[0] = self.outline_width // 2
        else:
            self.velocity[0] = velocity[0]

        if (velocity[1] > self.outline_height // 2):
            self.velocity[1] = self.outline_height // 2
        else:
            self.velocity[1] = velocity[1]

    def set_pos(self, new_pos):
        """
        self.pos の更新を行う。
        領域オーバーの場合はそれを補正し、さらに velocity を反転させてる。
        """
        if new_pos[0] > self.outline_width:
            new_pos[0] = 2 * self.outline_width - new_pos[0]
            self.set_velocity((self.velocity[0] * (-1), self.velocity[1]))
        elif new_pos[0] < 0:
            new_pos[0] = new_pos[0] * (-1)
            self.set_velocity((self.velocity[0] * (-1), self.velocity[1]))
        else:
            None  # do nothing

        if new_pos[1] > self.outline_height:
            new_pos[1] = 2 * self.outline_height - new_pos[1]
            self.set_velocity((self.velocity[0], self.velocity[1] * (-1)))
        elif new_pos[1] < 0:
            new_pos[1] = new_pos[1] * (-1)
            self.set_velocity((self.velocity[0], self.velocity[1] * (-1)))
        else:
            None  # do nothing

        self.pos[0] = new_pos[0]
        self.pos[1] = new_pos[1]

    def calc_next_pos(self):
        """
        次の pos を計算する。
        現在の pos に velocity を加算するだけ。
        計算結果が outline を超えていたら反射する
        """
        next_pos = [self.pos[0] + self.velocity[0],
                    self.pos[1] + self.velocity[1]]
        self.set_pos(next_pos)

    def get_pos(self):
        """
        計算用の座標系からプロット用の座標系に変換して出力。
        """
        return (self.pos[0] + self.radius, self.pos[1] + self.radius)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
