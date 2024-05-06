/*
    Component to display an error bar between the closest wire line segment endpoint and corridor
    line segment endpoint, scaled to the image height and width.
*/

interface Props {
    wireX: number;
    wireY: number; 
    corrX: number;
    corrY: number;
    clientWidth: number;
    clientHeight: number;
    visible: boolean;
}

function ErrorBar({ wireX, wireY, corrX, corrY, clientWidth, clientHeight, visible }: Props) {
    if (!visible) {
        return null;
    }
    const yaw = Math.atan2(corrY - wireY, corrX - wireX);
    const capLength = 10;

    const cap1sx = wireX + capLength * Math.cos(yaw + Math.PI / 2);
    const cap1sy = wireY + capLength * Math.sin(yaw + Math.PI / 2);
    const cap1ex = wireX + capLength * Math.cos(yaw - Math.PI / 2);
    const cap1ey = wireY + capLength * Math.sin(yaw - Math.PI / 2);

    const cap2sx = corrX + capLength * Math.cos(yaw + Math.PI / 2);
    const cap2sy = corrY + capLength * Math.sin(yaw + Math.PI / 2);
    const cap2ex = corrX + capLength * Math.cos(yaw - Math.PI / 2);
    const cap2ey = corrY + capLength * Math.sin(yaw - Math.PI / 2);

    const strokeOpacity = 0.6
    const strokeWidth = "4"
    return (
        <svg width={clientWidth} height={clientHeight} style={{ position: 'absolute', top: 0, left: 0 }}>
            <line
                x1={wireX}
                y1={wireY}
                x2={corrX}
                y2={corrY}
                stroke="red"
                strokeOpacity = {strokeOpacity}
                strokeWidth={strokeWidth}
            />
            <line
                x1={cap1sx}
                y1={cap1sy}
                x2={cap1ex}
                y2={cap1ey}
                stroke="red"
                strokeOpacity = {strokeOpacity}
                strokeWidth={strokeWidth}
            />
            <line
                x1={cap2sx}
                y1={cap2sy}
                x2={cap2ex}
                y2={cap2ey}
                stroke="red"
                strokeOpacity = {strokeOpacity}
                strokeWidth={strokeWidth}
            />
        </svg>
    );
}

export default ErrorBar;