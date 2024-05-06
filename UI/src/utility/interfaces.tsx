export interface Wire {
    coords: [[number, number], [number, number]];
}

interface Corridor  {
    relevant_bones: string[];
    coords: [[number, number], [number, number]];
}

interface Bone  {
    coords: Array<[number, number]>;
}

interface PoseData {
    corridors: { [key: string]: Corridor };
    bones: { [key: string]: Bone };
    wires: { [key: string]: Wire };
}
interface Image {
    enc: string;
    width: number;
    height: number;
}

export interface ServerData {
    img: Image;
    coords: {
        supine: PoseData;
        prone: PoseData;
    };
}

export interface ClosestPoints {
    wirePoint: [number, number];
    corridorPoint: [number, number];
}