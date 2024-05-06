/*
    Component to display a corridor on top of an X-ray image from a pair of 
    line segment endpoints.
*/

interface Props {
    x1: number;
    y1: number; 
    x2: number;
    y2: number;
}

// Function to obtain rectangle centered on line segment with given perpendicular thickness.
function rectFromLine(x1: number, y1: number, x2: number, y2: number, thickness: number) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.sqrt(dx * dx + dy * dy);
    const dirX = dx / length;
    const dirY = dy / length;

    const normalX = -dirY;
    const normalY = dirX;

    const offsetX = normalX * thickness / 2;
    const offsetY = normalY * thickness / 2;

    const points = [
        [x1 - offsetX, y1 - offsetY],
        [x2 - offsetX, y2 - offsetY],
        [x2 + offsetX, y2 + offsetY],
        [x1 + offsetX, y1 + offsetY]
    ];

    return points.map(point => point.join(',')).join(' ');
}

function Corridor({x1, y1, x2, y2}: Props) {
    const width = 10;
    const color = 'yellow';

    const pts_string = rectFromLine(x1, y1, x2, y2, width)

    return (
        <polygon
            fill={color}
            stroke={color}
            strokeWidth="2"
            points={pts_string}
            style={{ fillOpacity: 0.1, strokeOpacity: 0.3 }}
        />
    );
}

export default Corridor;
